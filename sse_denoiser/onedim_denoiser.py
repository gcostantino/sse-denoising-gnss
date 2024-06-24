import datetime
import os

import numpy as np
import pkbar
import torch
import torch_geometric

from utils import plot_actual_vs_predicted_cascadia, plot_disp_field_cascadia, \
    plot_actual_vs_predicted_cascadia_multiple, denoising_plots


class OneDimDenoiserModel(torch.nn.Module):
    """
    A one-dimensional denoising model using an LSTM and fully connected layers.

    Attributes:
        n_components (int): Number of components for the input features.
        rnn_hid_size (int): Hidden size of the LSTM.
        lin_emb_dim (int): Dimension of the linear embedding.
        bidirectional (bool): If True, use a bidirectional LSTM.
        lstm (torch.nn.LSTM): LSTM layer.
        relu (torch.nn.ReLU): ReLU activation function.
        lin1 (torch.nn.Linear): First linear layer.
        lin2 (torch.nn.Linear): Second linear layer.
    """

    def __init__(self):
        super(OneDimDenoiserModel, self).__init__()
        self.n_components = 2
        self.rnn_hid_size = 32  # from Xue and Freymueller's paper
        self.lin_emb_dim = 16
        self.bidirectional = True
        self.lstm = torch.nn.LSTM(input_size=self.n_components, hidden_size=self.rnn_hid_size,
                                  bidirectional=self.bidirectional, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(self.rnn_hid_size * 2 if self.bidirectional else self.rnn_hid_size,
                                    self.lin_emb_dim)
        self.lin2 = torch.nn.Linear(self.lin_emb_dim, self.n_components)

    def forward(self, x_in, visualize_rnn_feat=False, visualize_transf_feat=False):
        # only works when the tensor is 3D --> reshape
        n_batch, n_stations, window_length, n_dim = x_in.size()
        x_in_resh = torch.reshape(x_in, [n_batch * n_stations, window_length, n_dim])
        x, _ = self.lstm(x_in_resh)
        x = self.lin1(x)  # self.lin1(x[:, -1, :])  # changed wrt paper: output a sequence
        x = self.relu(x)
        x_out = self.lin2(x)
        x_out_resh = torch.reshape(x_out, [n_batch, n_stations, window_length, n_dim])  # --> back to 4D
        return x_out_resh


class OneDimensionalDenoiser:
    def __init__(self, **kwargs):
        self.model = None
        self.n_stations = kwargs['n_stations']
        self.window_length = kwargs['window_length']
        self.n_directions = kwargs['n_directions']
        self.batch_size = kwargs['batch_size']
        self.n_epochs = kwargs.pop('n_epochs', None)
        self.initial_learning_rate = kwargs.pop('learning_rate', None)
        self.kernel_regularizer = kwargs.pop('kernel_regularizer', None)
        self.bias_regularizer = kwargs.pop('bias_regularizer', None)
        self.embedding_regularizer = kwargs.pop('embedding_regularizer', None)
        # self.activation = tf.keras.layers.LeakyReLU(alpha=0.1)  # 'relu'
        # self.initializer = tf.keras.initializers.HeUniform()
        self.input_shape = (self.n_stations * self.window_length, self.n_directions)
        self.use_dropout = False
        self.dropout_at_test = False
        self.n_outputs = kwargs['n_outputs'] if 'n_outputs' in kwargs else 1
        self.use_batch_norm = kwargs['use_batch_norm'] if 'use_batch_norm' in kwargs else True
        self.callback_list = None
        self.train_verbosity_level = kwargs['verbosity'] if 'verbosity' in kwargs else 0
        self.patience = kwargs['patience'] if 'patience' in kwargs else 500
        self.station_coordinates = kwargs['station_coordinates'] if 'station_coordinates' in kwargs else None
        self.custom_loss = kwargs['custom_loss'] if 'custom_loss' in kwargs else False
        self.val_catalogue = kwargs['val_catalogue'] if 'val_catalogue' in kwargs else None
        self.amsgrad = kwargs['amsgrad'] if 'amsgrad' in kwargs else False
        self.loss = kwargs['loss'] if 'loss' in kwargs else 'mean_squared_error'
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.weight_path = None
        self.log_dir = None
        self.tb_writer = None
        self.y_val = kwargs.pop('y_val', None)
        self.scaler = kwargs.pop('scaler', None)
        self.device = None
        self.graph_loader = kwargs.pop('graph_loader', False)
        self.edge_index = kwargs.pop('edge_index', None)
        self.y_test = kwargs.pop('y_test', None)
        self.edge_attr = kwargs.pop('edge_attr', None)
        self.disp_learning = kwargs.pop('disp_learning', False)
        self.residual = kwargs.pop('residual', False)
        self.residual2 = kwargs.pop('residual2', False)
        self.residual3 = kwargs.pop('residual3', False)
        self.custom_loss_coeff = kwargs.pop('custom_loss_coeff', 1)
        self.return_attention = kwargs.pop('return_attention', False)

    def _callbacks(self, stagename):
        from torch.utils.tensorboard import SummaryWriter
        base_checkpoint_path, base_weight_dir = os.path.expandvars('$WORK') + '/models', os.path.expandvars(
            '$WORK') + '/weights'

        checkpoint_path = os.path.join(base_checkpoint_path, 'SSEdetector_char')
        weight_dir = os.path.join(base_weight_dir, 'SSEdetector_char')

        for path in [base_checkpoint_path, base_weight_dir, checkpoint_path, weight_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        # log_dir = "logs/fit/cascadia" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date_string = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
        log_dir = os.path.expandvars('$WORK') + "/logs/fit/cascadia" + date_string + f'_{stagename}'
        self.weight_path = os.path.join(weight_dir, f'best_cascadia_{date_string}_{stagename}.pt')

        self.tb_writer = SummaryWriter(log_dir)
        self.img_writer = SummaryWriter(os.path.join(log_dir, 'img'))

    def build(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = OneDimDenoiserModel()

        model.to(device)
        self.device = device
        self.model = model

    def summary_graph(self, x, e, b):
        print(torch_geometric.nn.summary(self.model, x=x.to(self.device), edge_index=e.to(self.device),
                                         batch=b.to(self.device), max_depth=1))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total params:', pytorch_total_params)
        print('Trainable params:', pytorch_trainable_params)
        print('Non-trainable params:', pytorch_total_params - pytorch_trainable_params)
        print('-' * 20)

    def summary_nograph(self, x):
        print(torch_geometric.nn.summary(self.model, x.to(self.device), max_depth=1))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total params:', pytorch_total_params)
        print('Trainable params:', pytorch_trainable_params)
        print('Non-trainable params:', pytorch_total_params - pytorch_trainable_params)
        print('-' * 20)

    def associate_optimizer(self):
        self.loss = torch.nn.MSELoss()  # torch.nn.HuberLoss()
        if self.residual3:
            self.noise_loss = torch.nn.MSELoss()
            self.signal_loss = torch.nn.MSELoss()
            self.loss = self.precision_loss

        if self.custom_loss:
            # self.loss = self.mw_disp_loss
            self.loss = self.ang_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.initial_learning_rate)

    def set_callbacks(self, train_codename):
        self._callbacks(train_codename)

    def set_data_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_epoch(self, progress_bar):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
            # Make predictions for this batch
            outputs = self.model(x, edge_index, batch)

            # Compute the loss and its gradients
            if not self.custom_loss:
                y = data[1].to(self.device)
                loss = self.loss(outputs, y)
            else:
                y1 = data[1].to(self.device)
                y_disp = data[2].to(self.device)
                loss = self.loss(outputs[0], y1, outputs[1], y_disp)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            progress_bar.update(i, values=[("loss: ", float(f'{loss.item():.4f}'))])

        # return last_loss
        return running_loss / len(self.train_loader)

    def minibatch_train(self):
        best_vloss = 1_000_000.
        for epoch in range(self.n_epochs):
            print("Epoch {}/{}".format(epoch + 1, self.n_epochs))
            progress_bar = pkbar.Kbar(target=len(self.train_loader), always_stateful=False, width=25,
                                      verbose=self.train_verbosity_level)
            self.model.train(True)
            avg_train_loss = self.train_epoch(progress_bar)
            progress_bar.add(1)
            # We don't need gradients on to do reporting
            self.model.eval()  # self.model.train(False)
            with torch.no_grad():
                running_vloss = 0.0
                end_of_epoch_val_pred = np.zeros(self.y_val.shape)
                n_val_batches = len(self.val_loader)
                for i, vdata in enumerate(self.val_loader):
                    # voutputs = model(vinputs)
                    voutputs = self.model(vdata.x.to(self.device), vdata.edge_index.to(self.device),
                                          vdata.batch.to(self.device))
                    vloss = self.loss(voutputs, vdata.y.to(self.device)).item()
                    running_vloss += vloss
                    # end_of_epoch_val_pred.append(voutputs.cpu().detach().numpy())
                    if i < n_val_batches - 1:
                        end_of_epoch_val_pred[
                        i * self.batch_size:(i + 1) * self.batch_size] = voutputs.cpu().detach().numpy()
                    else:
                        end_of_epoch_val_pred[i * self.batch_size:] = voutputs.cpu().detach().numpy()
                avg_vloss = running_vloss / (i + 1)

            # end_of_epoch_val_pred = np.array(end_of_epoch_val_pred)[0]

            # Log the running loss averaged per batch
            # for both training and validation
            self.tb_writer.add_scalars('',
                                       {'train': avg_train_loss, 'validation': avg_vloss}, epoch + 1)
            self.tb_writer.flush()
            if not self.disp_learning:
                y_val_scaled = self.scaler.inverse_transform(self.y_val)
                val_pred_scaled = self.scaler.inverse_transform(end_of_epoch_val_pred)
                if end_of_epoch_val_pred.shape[1] == 1:
                    figure_a_vs_p = plot_actual_vs_predicted_cascadia(y_val_scaled, val_pred_scaled, self.val_catalogue)
                else:
                    figure_a_vs_p = plot_actual_vs_predicted_cascadia_multiple(y_val_scaled, val_pred_scaled,
                                                                               self.val_catalogue)

                self.img_writer.add_figure('Act_vs_pred', figure_a_vs_p, epoch + 1, close=True)
            else:
                figure_disp = plot_disp_field_cascadia(self.y_val, end_of_epoch_val_pred,
                                                       self.val_catalogue, self.station_coordinates)
                self.img_writer.add_figure('displacements', figure_disp, epoch + 1, close=True)
            self.img_writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), self.weight_path)

            progress_bar.add(1,
                             values=[("loss: ", float(f'{avg_train_loss:.4f}')),
                                     ("val_loss: ", float(f'{avg_vloss:.4f}'))])

    def train_epoch_nograph(self, progress_bar):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            x = data[0].to(self.device)
            # Make predictions for this batch
            outputs = self.model(x)

            # Compute the loss and its gradients
            if not self.residual3:
                if not self.custom_loss:
                    y = data[1].to(self.device)
                    loss = self.loss(outputs, y)
                else:
                    '''y1 = data[1].to(self.device)
                    y_disp = data[2].to(self.device)
                    loss = self.loss(outputs[0], y1, outputs[1], y_disp)'''
                    y = data[1].to(self.device)
                    y_disp = (data[1][:, :, -1, :] - data[1][:, :, 0, :]).to(self.device)
                    pred_disp = (outputs[:, :, -1, :] - outputs[:, :, 0, :]).to(self.device)
                    loss = self.loss(y, outputs, y_disp, pred_disp)
            else:
                y_noise, y_sig = data[0].to(self.device) - data[1].to(self.device), data[1].to(self.device)
                noise_estim, signal_estim = outputs[0], outputs[1]
                loss = self.loss(y_noise, noise_estim, y_sig, signal_estim)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()

            progress_bar.update(i, values=[("loss: ", float(f'{loss.item():.4f}'))])

        return last_loss

    def minibatch_train_nograph(self):
        best_vloss = 1_000_000.
        for epoch in range(self.n_epochs):
            print("Epoch {}/{}".format(epoch + 1, self.n_epochs))
            progress_bar = pkbar.Kbar(target=len(self.train_loader), always_stateful=False, width=25,
                                      verbose=self.train_verbosity_level)
            self.model.train(True)
            avg_train_loss = self.train_epoch_nograph(progress_bar)
            progress_bar.add(1)
            # We don't need gradients on to do reporting
            self.model.eval()  # self.model.train(False)
            with torch.no_grad():
                running_vloss = 0.0
                end_of_epoch_val_pred = np.zeros(self.y_val.shape)
                n_val_batches = len(self.val_loader)
                for i, vdata in enumerate(self.val_loader):
                    # voutputs = model(vinputs)
                    voutputs = self.model(vdata[0].to(self.device))
                    if not self.residual3:
                        if not self.custom_loss:
                            vloss = self.loss(voutputs, vdata[1].to(self.device))
                        else:
                            # vloss = self.loss(voutputs[0], vdata[1].to(self.device), voutputs[1], vdata[2].to(self.device))
                            y = vdata[1].to(self.device)
                            y_disp = (vdata[1][:, :, -1, :] - vdata[1][:, :, 0, :]).to(self.device)
                            pred_disp = (voutputs[:, :, -1, :] - voutputs[:, :, 0, :]).to(self.device)
                            vloss = self.loss(y, voutputs, y_disp, pred_disp)
                    else:
                        y_noise, y_sig = vdata[0].to(self.device) - vdata[1].to(self.device), vdata[1].to(self.device)
                        noise_estim, signal_estim = voutputs[0], voutputs[1]
                        vloss = self.loss(y_noise, noise_estim, y_sig, signal_estim)
                        voutputs = signal_estim
                    running_vloss += vloss
                    # end_of_epoch_val_pred.append(voutputs.cpu().detach().numpy())
                    if not self.custom_loss:
                        if i < n_val_batches - 1:
                            end_of_epoch_val_pred[
                            i * self.batch_size:(i + 1) * self.batch_size] = voutputs.cpu().detach().numpy()
                        else:
                            end_of_epoch_val_pred[i * self.batch_size:] = voutputs.cpu().detach().numpy()
                    else:  # can be improved
                        if i < n_val_batches - 1:
                            end_of_epoch_val_pred[
                            i * self.batch_size:(i + 1) * self.batch_size] = voutputs[0].cpu().detach().numpy()
                        else:
                            end_of_epoch_val_pred[i * self.batch_size:] = voutputs[0].cpu().detach().numpy()

                avg_vloss = running_vloss / (i + 1)

            # end_of_epoch_val_pred = np.array(end_of_epoch_val_pred)[0]

            # Log the running loss averaged per batch
            # for both training and validation
            self.tb_writer.add_scalars('',
                                       {'train': avg_train_loss, 'validation': avg_vloss}, epoch + 1)
            self.tb_writer.flush()

            figure_disp = denoising_plots(self.val_loader, self.y_val, end_of_epoch_val_pred, self.val_catalogue,
                                          self.station_coordinates, static=self.learn_static)
            self.img_writer.add_figure('denoising', figure_disp, epoch + 1, close=True)
            self.img_writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), self.weight_path)

            progress_bar.add(1,
                             values=[("loss: ", float(f'{avg_train_loss:.4f}')),
                                     ("val_loss: ", float(f'{avg_vloss:.4f}'))])

    def _inference_nograph(self):
        test_pred = np.zeros(self.y_test.shape)
        if self.custom_loss:
            disp_f_pred = np.zeros((self.y_test.shape[0], self.n_stations, self.n_directions))
        n_test_batches = len(self.test_loader)
        self.model.eval()
        with torch.no_grad():
            for i, tdata in enumerate(self.test_loader):
                print(f'Batch {i + 1}/{n_test_batches}')
                toutputs = self.model(tdata[0].to(self.device))
                if self.custom_loss:
                    toutputs = toutputs[0]
                    dfoutput = toutputs[1]
                if self.residual3:
                    toutputs = toutputs[0]
                if i < n_test_batches - 1:
                    test_pred[i * self.batch_size:(i + 1) * self.batch_size] = toutputs.cpu().detach().numpy()
                    if self.custom_loss:
                        disp_f_pred[i * self.batch_size:(i + 1) * self.batch_size] = dfoutput.cpu().detach().numpy()
                else:
                    test_pred[i * self.batch_size:] = toutputs.cpu().detach().numpy()
                    if self.custom_loss:
                        disp_f_pred[i * self.batch_size:] = dfoutput.cpu().detach().numpy()
        if not self.custom_loss:
            return test_pred
        else:
            return test_pred, disp_f_pred

    def _inference(self):
        test_pred = np.zeros(self.y_test.shape)
        n_test_batches = len(self.test_loader)
        self.model.eval()
        with torch.no_grad():
            for i, tdata in enumerate(self.test_loader):
                print(f'Batch {i}/{n_test_batches}')
                toutputs = self.model(tdata.x.to(self.device), tdata.edge_index.to(self.device),
                                      tdata.batch.to(self.device))
                if i < n_test_batches - 1:
                    test_pred[i * self.batch_size:(i + 1) * self.batch_size] = toutputs.cpu().detach().numpy()
                else:
                    test_pred[i * self.batch_size:] = toutputs.cpu().detach().numpy()

        return test_pred

    def train(self):
        try:
            if self.graph_loader:
                self.minibatch_train()
            else:
                self.minibatch_train_nograph()
        except KeyboardInterrupt:
            print("Training of SSEdenoiser interrupted and completed")

    def load_weights(self, weight_path, by_name=False, strict=True):
        # self.model.load_weights(weight_path, by_name=by_name)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=strict)

    def inference(self):
        if self.graph_loader:
            return self._inference()
        else:
            return self._inference_nograph()

    def get_model(self):
        return self.model

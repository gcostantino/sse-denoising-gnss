import os
import sys

import joblib
from torch.utils.data import DataLoader as TorchDataLoader

from sse_denoiser.dataset_utils import Dataset
from sse_denoiser.stagrnn import STAGRNNDenoiser

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('Please provide CLI arguments.')
    n_samples, batch_size, learning_rate, work_directory = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), \
        sys.argv[4]

    set_callbacks = sys.argv[5].lower()
    set_callbacks = True if set_callbacks == 'true' else False

    add_transformer = True if sys.argv[6].lower() == 'true' else False
    use_spatial_attention = True if sys.argv[7].lower() == 'true' else False
    use_temporal_attention = True if sys.argv[8].lower() == 'true' else False

    verbosity = int(sys.argv[9]) if len(sys.argv) >= 10 else 1

    ablation_str = '_ablation'
    if not add_transformer:
        ablation_str += '_notransf'
    else:
        if use_temporal_attention:
            ablation_str += '_temp_att_only'
        if use_spatial_attention:
            ablation_str += '_spatial_att_only'

    train_codename = '...' + ablation_str

    dataset_filename = os.path.expandvars(
        work_directory) + '/denois_synth_ts_cascadia_realgaps_extended_v5_200stations_6_7_depth_20_40'

    data_dict = joblib.load(f'{dataset_filename}.data')

    data = data_dict['synthetic_data']
    durations = data_dict['random_durations']
    cat = data_dict['catalogue']
    static_displacement = data_dict['static_displacement']
    time_templates = data_dict['time_templates']
    station_codes = data_dict['station_codes']
    station_coordinates = data_dict['station_coordinates']

    y = time_templates[..., :2]

    y = y[:n_samples]
    data = data[:n_samples]
    cat = cat[:n_samples]

    n_stations = station_coordinates.shape[0]

    ind_val = int(n_samples * 0.8)
    ind_test = int(n_samples * 0.9)

    train_dataset = Dataset(data[:ind_val], y[:ind_val])
    val_dataset = Dataset(data[ind_val:ind_test], y[ind_val:ind_test])
    test_dataset = Dataset(data[ind_test:], y[ind_test:])

    cat_train, cat_val, cat_test = cat[:ind_val], cat[ind_val:ind_test], cat[ind_test:]
    '''templates_train, templates_val, templates_test = templates[:ind_val, :, :], templates[ind_val:ind_test, :,
                                                                                :], templates[ind_test:, :, :]'''

    y_train, y_val, y_test = y[:ind_val], y[ind_val:ind_test], y[ind_test:]

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    params = {'n_stations': n_stations,
              'window_length': 60,
              'n_directions': 2,
              'batch_size': batch_size,
              'n_epochs': 500,
              'learning_rate': learning_rate,
              'verbosity': verbosity,
              'patience': 500,
              'loss': 'mean_squared_error',
              'val_catalogue': cat_val,
              'station_coordinates': station_coordinates[:, :2],
              'y_val': y_val,
              'add_transformer': add_transformer,
              'use_temporal_attention': use_temporal_attention,
              'use_spatial_attention': use_spatial_attention}

    denoiser = STAGRNNDenoiser(**params)
    denoiser.build()
    denoiser.associate_optimizer()
    denoiser.set_data_loaders(train_loader, val_loader, test_loader)

    denoiser.summary_nograph(train_loader.dataset[:batch_size][0])

    if set_callbacks:
        denoiser.set_callbacks(train_codename)

    denoiser.train()

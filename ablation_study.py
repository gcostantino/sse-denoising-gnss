import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from prettytable import PrettyTable
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression

plt.rc('font', family='Helvetica')

SMALL_SIZE = 8 + 2 + 0
LEGEND_SIZE = 8 + 2 + 0
MEDIUM_SIZE = 10 + 2 + 0
BIGGER_SIZE = 12 + 2 + 0

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

models = ['SSEdenoiser', 'no_transformer', 'spatial_attention_only', 'temporal_attention_only']


# models = ['SSEdenoiser', 'no_transformer']


def signal_to_noise_ratio(signal, noise, target):
    snr = []
    for i in range(signal.shape[0]):
        valid_station_indices = np.where(target[i].sum(axis=(1, 2)) != 0)[0]  # stations with nonzero displacement
        if valid_station_indices.shape[0] != 0:
            valid_signal = signal[i, valid_station_indices, ...]
            valid_noise = noise[i, valid_station_indices, ...]
            # snr.append(np.mean(10 * np.log10(np.sum(valid_signal ** 2, axis=2) / np.sum(valid_noise ** 2, axis=2))))
            snr.append(np.mean(
                10 * np.log10(np.sum(valid_signal ** 2, axis=1) / np.sum(valid_noise ** 2, axis=1))))  # average on time
            ## snr.append(np.mean(10 * np.log10(peak_to_peak_signal ** 2 / peak_to_peak_noise ** 2)))
        else:
            snr.append(np.nan)
    return np.array(snr)


def binned_mae(label, prediction):
    window_length = label.shape[2]
    # return np.sum(np.abs(label - prediction)) / np.sum(extended_max_time_axis(label))
    static_offsets = np.abs(label[:, :, -1, :] - label[:, :, 0, :])
    # return np.sum(np.abs(label - prediction)) / np.sum(np.max(label, axis=2))
    return 1 / window_length * np.sum(np.abs(label - prediction)) / np.sum(np.max(np.abs(label), axis=2))
    per_sample_err = 1 / window_length * np.sum(np.abs(label - prediction), axis=(1, 2, 3)) / np.sum(
        np.max(np.abs(label), axis=2), axis=(1, 2))
    return 1 / window_length * np.sum(np.abs(label - prediction)) / np.sum(static_offsets)


def binned_cc(label, prediction):
    corrcoeffs = []
    n_obs = label.shape[0]
    if n_obs == 0:
        return np.nan
    for i in range(n_obs):
        cc = np.corrcoef(label[i].flatten(), prediction[i].flatten())
        corrcoeffs.append(cc[1, 0])
    return sum(corrcoeffs) / len(corrcoeffs)


def binned_static_disp_err(label, prediction):
    rel_errors = []
    delta_win = 7  # days
    n_obs = label.shape[0]
    if n_obs == 0:
        return np.nan

    '''static_disp_label = label[:, :, -1, :] - label[:, :, 0, :]
    static_disp_pred = prediction[:, :, -1, :] - prediction[:, :, 0, :]'''

    static_disp_label = np.median(label[:, :, -1 - delta_win:-1, :], axis=2) - np.median(label[:, :, 0:delta_win, :],
                                                                                         axis=2)
    static_disp_pred = np.median(prediction[:, :, -1 - delta_win:-1, :], axis=2) - np.median(
        prediction[:, :, 0:delta_win, :], axis=2)
    for i in range(n_obs):
        # rel_err = np.mean(np.abs(static_disp_label[i] - static_disp_pred[i]) / np.abs(static_disp_label[i]))
        rel_err = np.median(np.abs(static_disp_label[i] - static_disp_pred[i]) / np.abs(static_disp_label[i]))
        '''print('station 0, time step 0', np.abs(static_disp_label[i,0,0] - static_disp_pred[i,0,0]), "/", np.abs(static_disp_label[i, 0,0]))
        a, b = np.array([i] * 200*2), np.abs(static_disp_label[i].flatten() - static_disp_pred[i].flatten())/np.abs(static_disp_label[i].flatten())
        z = gaussian_kde(b)(b)
        idx = z.argsort()
        a, b, z = a[idx], b[idx], z[idx]
        plt.scatter(a, b, c=z)'''
        rel_errors.append(rel_err)
    # plt.show()
    # print('Number of points:', len(rel_errors))
    return sum(rel_errors) / len(rel_errors)


def binned_err_old(label, prediction, rmse=False):
    errors = []
    n_obs = label.shape[0]
    if n_obs == 0:
        return np.nan
    for i in range(n_obs):
        # err = np.mean((label[i]-prediction[i])**2)
        err = np.mean(np.abs(label[i] - prediction[i]))
        errors.append(err)
    final_err = sum(errors) / len(errors)
    return np.sqrt(final_err) if rmse else final_err


def binned_err(label, prediction):
    errors = []
    n_obs, _, n_time, _ = label.shape
    if n_obs == 0:
        return np.nan
    for i in range(n_obs):
        valid_station_indices = np.where(label[i].sum(axis=(1, 2)) != 0)[0]  # stations with nonzero displacement
        static_disp_label = label[i][valid_station_indices, -1, :] - label[i][valid_station_indices, 0, :]
        abs_err = np.mean(np.abs(label[i][valid_station_indices, ...] - prediction[i][valid_station_indices, ...]),
                          axis=1)
        inner_err = abs_err / np.max(np.abs(label[i][valid_station_indices, ...]), axis=1)
        # inner_err = abs_err / np.abs(static_disp_label)
        err = np.mean(inner_err)  # err = 1 / n_time * np.mean(inner_err)
        errors.append(err)
    return np.mean(errors), np.std(errors)


def mae_as_function_of_snr(y_true, y_pred, x, N_BINS=20):
    valid_values = ~np.isnan(x)
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]
    _, bin_edges = np.histogram(x, bins=N_BINS)
    snr_bin_mean = []
    err_bin_param = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        err_bin_param.append(binned_mae(y_true[idx_bin], y_pred[idx_bin]))
        snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    return snr_bin_mean, err_bin_param


def as_function_of(x, y, N_BINS=20):
    # valid_values = ~np.isnan(x)
    # x, y = x[valid_values], y[valid_values]
    _, bin_edges = np.histogram(x, bins=N_BINS)
    binned_mean = []
    binned_param = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        binned_param.append(np.mean(y[idx_bin]))
        binned_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    return binned_param, binned_mean


def cc_as_function_of_snr(y_true, y_pred, x, N_BINS=20):
    valid_values = ~np.isnan(x)
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]
    _, bin_edges = np.histogram(x, bins=N_BINS)
    snr_bin_mean = []
    err_bin_param = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        err_bin_param.append(binned_cc(y_true[idx_bin], y_pred[idx_bin]))
        snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    return snr_bin_mean, err_bin_param


def static_disp_as_function_of_snr(y_true, y_pred, x, N_BINS=20):
    valid_values = np.logical_and(~np.isnan(x), ~np.isinf(x))  # ~np.isnan(x)
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]
    _, bin_edges = np.histogram(x, bins=N_BINS)
    snr_bin_mean = []
    err_bin_param = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        err_bin_param.append(binned_static_disp_err(y_true[idx_bin], y_pred[idx_bin]))
        snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    return snr_bin_mean, err_bin_param


def err_as_function_of_snr(y_true, y_pred, x, N_BINS=20):
    valid_values = np.logical_and(~np.isnan(x), ~np.isinf(x))  # ~np.isnan(x)
    x, y_true, y_pred = x[valid_values], y_true[valid_values], y_pred[valid_values]
    _, bin_edges = np.histogram(x, bins=N_BINS)
    snr_bin_mean = []
    err_bin_param = []
    err_std = []
    for i in range(len(bin_edges) - 1):
        idx_bin = np.where(np.logical_and(x >= bin_edges[i], x < bin_edges[i + 1]))[0]
        print(f'Number of elements in bin {i}:', len(idx_bin))
        mean_bin, std_bin = binned_err(y_true[idx_bin], y_pred[idx_bin])
        err_bin_param.append(mean_bin)
        err_std.append(std_bin)
        snr_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    snr_bin_mean, err_bin_param, err_std = np.array(snr_bin_mean), np.array(err_bin_param), np.array(std_bin)
    return snr_bin_mean, err_bin_param, err_std


def _moving_median(data, window_size=3):
    moving_median = medfilt(data, kernel_size=(1, 1, window_size, 1))
    return moving_median


def _moving_mean(data, window_size=3, time_axis=2):
    filt = np.ones(window_size) / window_size
    # moving_mean = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=time_axis, arr=data)
    # moving_mean = scipy.ndimage.convolve1d(data, filt, axis=time_axis)
    moving_mean = scipy.ndimage.uniform_filter1d(data, window_size, axis=time_axis)
    return moving_mean


def _rmse_4d(original, predicted):
    m, n, p, q = original.shape  # m: n_samples, n: n_stations, q: n_time_steps, q: n_directions
    mse = np.sum((original - predicted) ** 2) / (m * n * p * q)
    rmse = np.sqrt(mse)
    return rmse


def _mae_4d(original, predicted):
    m, n, p, q = original.shape  # m: n_samples, n: n_stations, q: n_time_steps, q: n_directions
    mae = np.sum(np.abs(original - predicted)) / (m * n * p * q)
    return mae


def _squared_error_3d(original, predicted):
    se = np.mean((original - predicted) ** 2, axis=(1, 2, 3))  # the mean over the other axes is taken beforehand
    return se


def _absolute_error_3d(original, predicted):
    ae = np.mean(np.abs(original - predicted), axis=(1, 2, 3))  # the mean over the other axes is taken beforehand
    return ae


def _load_synth_data(filename, load_pred_only=False):
    data_dict = joblib.load(filename)
    pred = data_dict['pred']
    if load_pred_only:
        return pred
    X = data_dict['X']
    y = data_dict['y']
    catalogue = data_dict['cat']
    return X, y, catalogue, pred


def load_all_synth_data(extended=False):
    ext_str = '_extended' if extended else ''
    X, y, catalogue, pred_notransf = _load_synth_data(
        'predictions/ablation/pred_denoising_test_data_ablation_notransf' + ext_str)
    pred_ssedenoiser = _load_synth_data('predictions/pred_denoising_test_data' + ext_str, load_pred_only=True)
    pred_spatial_only = _load_synth_data(
        'predictions/ablation/pred_denoising_test_data_ablation_spatial_att_only' + ext_str, load_pred_only=True)
    pred_temporal_only = _load_synth_data(
        'predictions/ablation/pred_denoising_test_data_ablation_temp_att_only' + ext_str, load_pred_only=True)
    return X, y, catalogue, [pred_ssedenoiser, pred_notransf, pred_spatial_only, pred_temporal_only]


def compute_mean_median_predictions(data, kernel_values, extended=False, save=False,
                                    save_directory='pred_denoising_test_data_mean_median_3_7_15'):
    """Format: mean_kernel1, median_kernel1, mean_kernel2, median_kernel2, ..., mean_kernel_k, median_kernel_k."""
    ext_str = '_extended' if extended else ''
    predictions = []
    for kernel in kernel_values:
        predictions.append(_moving_mean(data, kernel))
        predictions.append(_moving_median(data, kernel))
    if save:
        data_dict = dict()
        data_dict['pred'] = predictions
        joblib.dump(data_dict, save_directory + ext_str)
    else:
        return predictions


def load_mean_median_predictions(extended=False):
    ext_str = '_extended' if extended else ''
    data_dict = joblib.load('pred_denoising_test_data_mean_median_3_7_15' + ext_str)
    pred = data_dict['pred']
    return pred


def compute_rmse_predictions(labels, prediction_list, mse=False):
    error_list = []
    for prediction in prediction_list:
        error = _rmse_4d(labels, prediction)
        error_list.append(error ** 2 if mse else error)
    return error_list


def compute_mae_predictions(labels, prediction_list):
    error_list = []
    for prediction in prediction_list:
        error = _mae_4d(labels, prediction)
        error_list.append(error)
    return error_list


def compute_squared_error_predictions(labels, prediction_list):
    error_list = []
    for prediction in prediction_list:
        error = _squared_error_3d(labels, prediction)
        error_list.append(error)
    return error_list


def compute_absolute_error_predictions(labels, prediction_list):
    error_list = []
    for prediction in prediction_list:
        error = _absolute_error_3d(labels, prediction)
        error_list.append(error)
    return error_list


def statistics_table(data, labels, predictions, save=False):
    '''rmse_errors = compute_rmse_predictions(labels, predictions, mse=False)
    mae_errors = compute_mae_predictions(labels, predictions)
    print(rmse_errors)
    print(mae_errors)'''

    squared_errors = compute_squared_error_predictions(labels, predictions)
    absolute_errors = compute_absolute_error_predictions(labels, predictions)

    table = PrettyTable(['Model', 'MSE', 'MAE'])
    for i, model in enumerate(models):
        table.add_row([model, f'{squared_errors[i].mean():.2f}', f'{absolute_errors[i].mean():.2f}'])
    print(table)

    table = PrettyTable(['Model', 'MSE ± std_se', 'MAE ± std_ae'])
    for i, model in enumerate(models):
        table.add_row([model, f'{squared_errors[i].mean():.2f} ± {squared_errors[i].std():.2f}',
                       f'{absolute_errors[i].mean():.2f} ± {absolute_errors[i].std():.2f}'])
    print(table)

    if save:
        with open('statistics_ablation.txt', 'w') as f:
            f.write(table.__str__())


def error_vs_snr(data, labels, predictions):
    # for the moment, 0-dislocation samples are taken into account
    snr = signal_to_noise_ratio(data, data - labels, labels)

    x, y = mae_as_function_of_snr(labels, predictions[-1], snr)
    plt.plot(x, y)
    plt.ylabel('Average denoising error [mm]')
    plt.xlabel('signal-to-noise ratio [dB]')
    plt.show()


def err_vs_snr(data, labels, predictions):
    # for the moment, 0-dislocation samples are not taken into account
    snr = signal_to_noise_ratio(data, data - labels, labels)
    valid_snr_idx = snr > 0
    snr = snr[valid_snr_idx]
    for i in range(len(predictions)):
        x, y, std = err_as_function_of_snr(labels[valid_snr_idx], predictions[i][valid_snr_idx], snr, N_BINS=10)
        plt.plot(x, y, marker='o', label=models[i])
        # plt.fill_between(x, y - std, y + std, alpha=0.2)
    plt.ylabel('Average relative error')
    plt.xlabel('signal-to-noise ratio [dB]')
    plt.yscale('log')
    plt.legend()
    plt.savefig('ablation_plots/err_vs_snr_log_ablation.pdf', bbox_inches='tight')
    # plt.show()


def err_vs_disp(y_true, y_pred):
    component = 0
    model_colors = ['C0', 'C1', 'C2']
    '''for n_models in range(len(y_pred)):
        for n_stations in range(y_true.shape[1]):
            static_disp_true = y_true[:, n_stations, -1, component] - y_true[:, n_stations, 0, component]
            static_disp_pred = y_pred[n_models][:, n_stations, -1, component] - y_pred[n_models][:, n_stations, 0, component]
            x = np.abs(static_disp_true)
            y = np.abs(static_disp_true - static_disp_pred)
            x_bin, y_bin = as_function_of(x, y, N_BINS=5)
            plt.scatter(x_bin, y_bin, s=10, c=model_colors[n_models])
        plt.scatter([], [], c=model_colors[n_models], label=models[6 + n_models])'''
    '''for n_models in range(len(y_pred)):
        static_disp_true = y_true[:, :, -1, component] - y_true[:, :, 0, component]
        static_disp_pred = y_pred[n_models][:, :, -1, component] - y_pred[n_models][:, :, 0, component]
        x = np.abs(static_disp_true)
        y = np.abs(static_disp_true - static_disp_pred)
        x_bin, y_bin = as_function_of(x.ravel(), y.ravel(), N_BINS=10)
        print(np.array(x_bin).shape, np.array(y_bin).shape)
        idx_sort = np.argsort(x_bin)
        print(x_bin, y_bin)
        x_bin, y_bin = np.array(x_bin), np.array(y_bin)
        plt.plot(x_bin[idx_sort], y_bin[idx_sort], c=model_colors[n_models])

        plt.scatter([], [], c=model_colors[n_models], label=models[6 + n_models])'''
    cmap_models = ['Blues', 'Reds', 'Oranges']
    station = 0
    eps = 1e-03
    for n_models in range(len(y_pred)):
        # static_disp_true = y_true[:, :, -1, component] - y_true[:, :, 0, component]
        # static_disp_pred = y_pred[n_models][:, :, -1, component] - y_pred[n_models][:, :, 0, component]
        # static_disp_true = np.min(y_true[:, :, :, component], axis=-1)
        # static_disp_pred = np.min(y_pred[n_models][:, :, :, component], axis=-1)
        static_disp_true = y_true[:, :, :, component]
        static_disp_pred = y_pred[n_models][:, :, :, component]
        # x = np.abs(static_disp_true)
        x = np.abs(static_disp_pred)  # np.abs(static_disp_true)
        y = np.abs(static_disp_true / (eps + static_disp_pred))  # np.abs(static_disp_true - static_disp_pred)
        # x_bin, y_bin = as_function_of(x.ravel(), y.ravel(), N_BINS=10)
        # idx_sort = np.argsort(x_bin)
        # x_bin, y_bin = np.array(x_bin), np.array(y_bin)
        # plt.plot(x_bin[idx_sort], y_bin[idx_sort], c=model_colors[n_models])

        # x, y = x[:, station], y[:, station]
        x, y = x[:, station, :], y[:, station, :]
        x, y = x.ravel(), y.ravel()

        '''# kernel density estimate
        xmin, xmax = 0, 100
        ymin, ymax = 0, 100
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        cfset = plt.contourf(xx, yy, f, cmap=cmap_models[n_models])
        cset = plt.contour(xx, yy, f, colors='k')'''

        '''idx_sort = np.argsort(x)
        x, y = np.array(x), np.array(y)
        plt.scatter(x[idx_sort], y[idx_sort], c=model_colors[n_models])'''

        static_disp_pred = static_disp_pred.ravel()
        static_disp_true = static_disp_true.ravel()

        model = LinearRegression()
        model.fit(static_disp_pred.reshape(-1, 1), static_disp_true.reshape(-1, 1))
        # Extract the slope (coefficient) of the linear regression model
        c = model.coef_[0]

        print("Value of c:", c, "intercept:", model.intercept_[0])
        plt.scatter(static_disp_pred, static_disp_true, s=5, c=model_colors[n_models])
        plt.plot(static_disp_pred, model.coef_[0] * static_disp_pred + model.intercept_[0], linewidth=2)
        # x_bin, y_bin = as_function_of(x.ravel(), y.ravel(), N_BINS=5)
        x_bin, y_bin = as_function_of(static_disp_pred, static_disp_true, N_BINS=5)
        idx_sort = np.argsort(x_bin)
        x_bin, y_bin = np.array(x_bin), np.array(y_bin)
        plt.plot(x_bin[idx_sort], y_bin[idx_sort], c=model_colors[n_models])

        plt.scatter([], [], c=model_colors[n_models], label=models[6 + n_models])
    plt.legend()
    plt.title('E-W error')
    plt.xlabel('Static displacement [mm]')
    plt.ylabel('Mean absolute error on static displacement [mm]')
    plt.show()


if __name__ == '__main__':
    extended_dataset = False
    print('Loading data...')
    X, y, catalogue, predictions = load_all_synth_data(extended=extended_dataset)
    print('Data loaded')

    statistics_table(X, y, predictions, save=True)

    err_vs_snr(X, y, predictions)

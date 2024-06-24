import os

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from sse_denoiser.dataset_utils import Dataset
from sse_denoiser.stagrnn import STAGRNNDenoiser
from utils import _preliminary_operations, cascadia_filtered_stations, detrend_nan_v2

if __name__ == '__main__':
    reference_period = (2007, 2023)
    n_selected_stations = 200
    window_length = 60
    work_directory = '$WORK'
    batch_size = 32
    n_directions = 2
    learn_static = False
    residual = False
    residual2 = False
    residual3 = False
    new_stress_drop = False

    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False)
    station_codes, station_coordinates, full_station_codes, full_station_coordinates, station_subset = cascadia_filtered_stations(
        n_selected_stations)
    selected_gnss_data = selected_gnss_data[station_subset]
    original_nan_pattern = np.isnan(selected_gnss_data[:, :, 0])
    selected_gnss_data, trend_info = detrend_nan_v2(selected_time_array, selected_gnss_data)

    selected_gnss_data[original_nan_pattern] = 0.  # NaNs are replaced with zeros

    running_test_set = []
    if learn_static:
        for i in range(window_length // 2, selected_time_array.shape[0] - window_length // 2):
            data_window = selected_gnss_data[:, i - window_length // 2:i + window_length // 2, :]
            running_test_set.append(data_window)
    else:
        for i in range(selected_time_array.shape[0] - window_length):
            data_window = selected_gnss_data[:, i:i + window_length, :]
            running_test_set.append(data_window)
    running_test_set = np.array(running_test_set)

    if learn_static:
        dummy_y_test = torch.Tensor(np.zeros((running_test_set.shape[0], n_selected_stations, n_directions)))
    else:
        dummy_y_test = torch.Tensor(
            np.zeros((running_test_set.shape[0], n_selected_stations, window_length, n_directions)))
    test_dataset = Dataset(running_test_set, dummy_y_test)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    params = {'n_stations': n_selected_stations,
              'window_length': window_length,
              'n_directions': n_directions,
              'batch_size': batch_size,
              'station_coordinates': station_coordinates[:, :2],
              'y_test': dummy_y_test,
              'learn_static': learn_static,
              'residual': residual,
              'residual2': residual2}

    denoiser = STAGRNNDenoiser(**params)
    denoiser.build()

    denoiser.set_data_loaders(None, None, test_loader)

    weight_path = os.path.join(os.path.expandvars(work_directory),
                               'weights/best_cascadia_02Jul2023-013725_train_denois_realgaps_v5_STAGRNN_no_CNN_old_sd.pt')

    denoiser.load_weights(weight_path)

    pred = denoiser.inference()
    np.savez(os.path.join(os.path.expandvars(work_directory), 'pred_SSEdenoiser'), pred=pred)

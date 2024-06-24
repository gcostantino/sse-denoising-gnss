import os
import sys

import joblib
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader

from sse_denoiser.dataset_utils import Dataset
from sse_denoiser.twodim_denoiser import TwoDimensionalDenoiser

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('Please provide CLI arguments.')
    n_samples, batch_size, work_directory = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

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

    n_stations = station_coordinates.shape[0]

    y = y[:n_samples]
    data = data[:n_samples]
    cat = cat[:n_samples]

    # data is sorted by latitude
    latsort = np.argsort(station_coordinates[:, 0])  # from greater/lower is not important
    data = data[:, latsort, :, :]
    y = y[:, latsort, :]
    station_coordinates = station_coordinates[latsort, :2]  # latitude and longitude only

    # data is padded for the U-Net
    data = np.pad(data, ((0, 0), (4, 4), (2, 2), (0, 0)))
    y = np.pad(y, ((0, 0), (4, 4), (2, 2), (0, 0)))

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
              'learning_rate': 0.001,
              'y_test': y_test,
              'verbosity': 1,
              'station_coordinates': station_coordinates[:, :2]}

    denoiser = TwoDimensionalDenoiser(**params)
    denoiser.build()
    denoiser.associate_optimizer()
    denoiser.set_data_loaders(train_loader, val_loader, test_loader)

    denoiser.summary_nograph(test_loader.dataset[:batch_size][0])

    weight_path = os.path.join(os.path.expandvars(work_directory),
                               'weights/best_cascadia_10Jan2024-221036_train_denois_realgaps_v5_TwoDimDenoiserUnetLatSort.pt')

    denoiser.load_weights(weight_path)

    pred = denoiser.inference()

    # data is cropped and got back to the original sorting
    data = data[:, 4:-4, 2:-2, :]
    y_test = y_test[:, 4:-4, 2:-2, :]
    pred = pred[:, 4:-4, 2:-2, :]

    original_sorting = np.argsort(latsort)  # from greater/lower is not important
    data = data[:, original_sorting, :, :]
    y_test = y_test[:, original_sorting, :]
    pred = pred[:, original_sorting, :]
    # station_coordinates = station_coordinates[original_sorting, :2]

    dataset_path = os.path.expandvars(work_directory) + '/pred_denoising_test_data_2d_unet'
    data_dict = dict()
    data_dict['pred'] = pred
    data_dict['X'] = data[ind_test:]
    data_dict['y'] = y_test
    data_dict['cat'] = cat_test

    joblib.dump(data_dict, dataset_path)

    # np.savez(os.path.expandvars(work_directory) + '/pred_denoising_test_data', pred=pred, X=data[ind_test:], y=y_test, cat=cat_test)

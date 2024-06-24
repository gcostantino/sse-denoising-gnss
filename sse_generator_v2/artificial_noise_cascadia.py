"""
Python script used to generate artificial noise samples for the Cascadia subduction zone
"""
import math
import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from sse_generator_v2.surrogates import Surrogates
from utils import cascadia_coordinates, _preliminary_operations, _remove_stations, detrend_nan_v2, \
    cascadia_filtered_stations


def n_surrogates_stack(data, n_surrogates, n_iterations=5, parallel=False):
    """Generator of n_surrogates from the original data passed as input.
    It exploits the caching of the precomputed Fourier Transforms."""
    try:
        np.isfinite(data).all()
    except Exception('Fatal error. Data is not finite.'):
        exit(-1)
    output_stack = np.zeros((n_surrogates, data.shape[0], data.shape[1]))
    surr = Surrogates(data, silence_level=2)
    if not parallel:
        for i in range(n_surrogates):
            output_stack[i] = surr.refined_AAFT_surrogates(data, n_iterations=n_iterations, output='true_amplitudes')
    else:
        for i, sample in enumerate(Parallel(n_jobs=multiprocessing.cpu_count(), verbose=True)(
                delayed(surr.refined_AAFT_surrogates)(data, n_iterations=n_iterations, output='true_amplitudes') for _
                in range(n_surrogates))):
            output_stack[i] = sample
    return output_stack


def n_surrogates_stack_pca_lib(data, n_surrogates, n_iterations=5, parallel=False):
    """Function performing n surrogate data generations by computing first a PCA,
    a surrogate generation (Schreiber et al., 2000) and an inverse PCA.
    It exploits the caching of the Fourier Transforms."""
    sample = data.T  # columns (features) -> stations
    pca = PCA(n_components=sample.shape[1])
    rotated_sample = pca.fit_transform(sample)
    rotated_surr = n_surrogates_stack(rotated_sample.T, n_surrogates, n_iterations=n_iterations, parallel=parallel)
    output_surrogates = np.zeros(rotated_surr.shape)
    for i in range(output_surrogates.shape[0]):
        output_surrogates[i] = pca.inverse_transform(rotated_surr[i].T).T
    return output_surrogates


def artificial_noise_full_randomized_v2(n, n_selected_stations, window_length, reference_period, p=0.7,
                                        add_trend=False):
    """As artificial_noise_full_randomized(), but it adds random data gaps, with the following changes:
        - before any pre-processing, the following stations are removed, either because they have outlier values,
            or because the percentage of useful data in the reference period is ~0:
            - ['WSLB', 'YBHB', 'P687', 'BELI', 'PMAR', 'TGUA', 'OYLR', 'FTS5', 'RPT5', 'RPT6', 'P791', 'P674',
                'P656', 'TWRI', 'WIFR', 'FRID', 'PNHG', 'COUR', 'SKMA', 'CSHR', 'HGP1', 'CBLV', 'PNHR', 'NCS2',
                'TSEP', 'BCSC']
        - data is detrended after choosing the subset of n_selected_stations.
        - instead of forcing data gaps to be equal to zero, NaNs values are enforced, which leads to a proper
            behaviour when synthetic SSEs are added into the window.
    """
    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False)
    station_codes, station_coordinates = cascadia_coordinates()
    stations_to_remove = ['WSLB', 'YBHB', 'P687', 'BELI', 'PMAR', 'TGUA', 'OYLR', 'FTS5', 'RPT5', 'RPT6', 'P791',
                          'P674', 'P656', 'TWRI', 'WIFR', 'FRID', 'PNHG', 'COUR', 'SKMA', 'CSHR', 'HGP1', 'CBLV',
                          'PNHR', 'NCS2', 'TSEP', 'BCSC']
    station_codes, station_coordinates, selected_gnss_data = _remove_stations(stations_to_remove, station_codes,
                                                                              station_coordinates, selected_gnss_data)
    original_nan_pattern = np.isnan(selected_gnss_data[:, :, 0])
    n_nans_stations = original_nan_pattern.sum(axis=1)
    stations_subset = np.sort(np.argsort(n_nans_stations)[:n_selected_stations])
    station_codes_subset, station_coordinates_subset = np.array(station_codes)[stations_subset], station_coordinates[
                                                                                                 stations_subset, :]
    original_nan_pattern = original_nan_pattern[stations_subset, :]
    gnss_data_subset = selected_gnss_data[stations_subset, :, :]
    gnss_data_subset, trend_info = detrend_nan_v2(selected_time_array, gnss_data_subset)
    gnss_data_subset[original_nan_pattern] = 0.  # NaNs are replaced with zeros
    trends = np.repeat(selected_time_array.reshape(-1, 1), n_selected_stations, axis=1) * trend_info[:, 0,
                                                                                          0] + trend_info[:, 0, 1]
    if not add_trend:
        trends.fill(0.)
    n_non_overlapping_windows = selected_time_array.shape[0] // window_length
    n_surrogates = math.ceil(n / n_non_overlapping_windows)
    # output_data = np.zeros((n, gnss_data_subset.shape[0], window_length, gnss_data_subset.shape[2]))
    output_data = np.zeros(
        (n_non_overlapping_windows * n_surrogates, gnss_data_subset.shape[0], window_length, gnss_data_subset.shape[2]))
    for direction in range(gnss_data_subset.shape[2]):
        surrogates = n_surrogates_stack_pca_lib(gnss_data_subset[:, :, direction], n_surrogates, n_iterations=5,
                                                parallel=True)
        for sur in range(n_surrogates):
            if np.random.random() > 1 - p:
                perm = np.random.permutation(original_nan_pattern.shape[0])
                permuted_pattern = original_nan_pattern[perm]
                # surrogates[:, original_nan_pattern] = 0.  # original NaNs are restored
                # surrogates[sur, permuted_pattern] = 0.
                surrogates[sur, permuted_pattern] = np.nan
        surrogates = np.roll(surrogates, np.random.randint(-window_length, window_length), axis=2)
        for i in range(n_surrogates):
            for j in range(n_non_overlapping_windows):
                output_data[i * n_non_overlapping_windows + j, :, :, direction] = surrogates[i, :,
                                                                                  j * window_length:(
                                                                                                            j + 1) * window_length]
    return output_data[:n], station_codes_subset, station_coordinates_subset


def artificial_noise_full_randomized_v3(n, n_selected_stations, window_length, reference_period, p=0.7,
                                        add_trend=False):
    """As artificial_noise_full_randomized_v2(), but we perform the station selection on the whole (2007,2023)
    period. Another station is removed: 'LNG2', through the cascadia_filtered_stations() function.
    """
    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False)
    station_codes_subset, station_coordinates_subset, full_station_codes, full_station_coordinates, station_subset = cascadia_filtered_stations(
        n_selected_stations)
    gnss_data_subset = selected_gnss_data[station_subset]

    original_nan_pattern = np.isnan(gnss_data_subset[:, :, 0])

    gnss_data_subset, trend_info = detrend_nan_v2(selected_time_array, gnss_data_subset)
    gnss_data_subset[original_nan_pattern] = 0.  # NaNs are replaced with zeros
    trends = selected_time_array[np.newaxis, :, np.newaxis] * trend_info[:, :, 0][:, np.newaxis, :] + trend_info[:, :,
                                                                                                      1][:, np.newaxis,
                                                                                                      :]

    n_non_overlapping_windows = selected_time_array.shape[0] // window_length
    n_surrogates = math.ceil(n / n_non_overlapping_windows)
    # output_data = np.zeros((n, gnss_data_subset.shape[0], window_length, gnss_data_subset.shape[2]))
    output_data = np.zeros(
        (n_non_overlapping_windows * n_surrogates, gnss_data_subset.shape[0], window_length, gnss_data_subset.shape[2]))
    for direction in range(gnss_data_subset.shape[2]):
        surrogates = n_surrogates_stack_pca_lib(gnss_data_subset[:, :, direction], n_surrogates, n_iterations=5,
                                                parallel=True)
        if add_trend:
            surrogates = surrogates + trends[np.newaxis, :, :, direction]
        for sur in range(n_surrogates):
            if np.random.random() > 1 - p:
                perm = np.random.permutation(original_nan_pattern.shape[0])
                permuted_pattern = original_nan_pattern[perm]
                # surrogates[:, original_nan_pattern] = 0.  # original NaNs are restored
                # surrogates[sur, permuted_pattern] = 0.
                surrogates[sur, permuted_pattern] = np.nan
        surrogates = np.roll(surrogates, np.random.randint(-window_length, window_length), axis=2)
        for i in range(n_surrogates):
            for j in range(n_non_overlapping_windows):
                output_data[i * n_non_overlapping_windows + j, :, :, direction] = surrogates[i, :,
                                                                                  j * window_length:(
                                                                                                            j + 1) * window_length]
    return output_data[:n], station_codes_subset, station_coordinates_subset

import os
import sys

import joblib

from sse_generator_v2.synthetic_time_series_cascadia import synthetic_time_series_real_gaps_extended_v5


def generate_v5_original_data():
    from data_config_files import original_v5_parameters as params_original

    if len(sys.argv) <= 1:
        raise Exception('Please provide CLI arguments.')
    n_samples = int(sys.argv[1])

    n_stations = params_original.n_stations
    window_length = params_original.window_length
    magnitude_range = params_original.magnitude_range
    depth_range = params_original.depth_range
    rake_range = params_original.rake_range
    data_gap_proba = params_original.data_gap_proba
    new_stress_drop = params_original.data_gap_proba

    dset_type = 'denois'
    base_dir = os.path.expandvars('$WORK')  # '.'

    dataset_filename = f'{dset_type}_synth_ts_cascadia_realgaps_extended_v5_{n_stations}stations_{magnitude_range[0]}_{magnitude_range[1]}_depth_{depth_range[0]}_{depth_range[1]}.data'

    dataset_path = os.path.join(base_dir, dataset_filename)

    synth_data, rand_dur, cat, synth_disp, templ, stat_codes, stat_coord = synthetic_time_series_real_gaps_extended_v5(
        n_samples, n_stations, window_length=window_length, magnitude_range=magnitude_range,
        depth_range=depth_range, rake_range=rake_range, p=data_gap_proba, new_stress_drop=new_stress_drop)

    #################################################################################
    '''import matplotlib.pyplot as plt
    import numpy as np

    latsort = np.argsort(stat_coord[:, 0])
    for i in range(templ.shape[0]):
        fig, axes = plt.subplots(1, 2)
        ms1 = axes[0].matshow(synth_data[i][latsort, :, 0], vmin=3, vmax=-5)
        ms2 = axes[1].matshow(templ[i][latsort, :, 0], vmin=3, vmax=-5)
        plt.colorbar(ms1, ax=axes[0])
        plt.colorbar(ms2, ax=axes[1])
        plt.show()'''
    #################################################################################

    data_dict = dict()
    data_dict['synthetic_data'] = synth_data
    data_dict['random_durations'] = rand_dur
    data_dict['catalogue'] = cat
    data_dict['static_displacement'] = synth_disp
    data_dict['time_templates'] = templ
    data_dict['station_codes'] = stat_codes
    data_dict['station_coordinates'] = stat_coord

    joblib.dump(data_dict, dataset_path)


if __name__ == '__main__':
    generate_v5_original_data()

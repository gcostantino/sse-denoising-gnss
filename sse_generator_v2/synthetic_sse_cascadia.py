"""
Script used to model a synthetic slow slip event in the Cascadia subduction zone.
"""
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from utils import read_from_slab2, compute_geodesic_km_conversion_array
from .okada import forward as okada85


def _synthetic_displacement_stations_cascadia_v2(i, depth_list, strike_list, dip_list, station_coordinates, **params):
    """Same as '_synthetic_displacement_stations_cascadia' (SSEdetector). What's new:
    - more dislocations can be associated to the same sample. We return them as a list."""
    n_dislocations = np.random.randint(low=0, high=1 + params['max_n_disloc'])  # zero to 3 dislocations (default)
    displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, Mw_all, strike_all, dip_all, rake_all, u_all, stress_drop_all = [], [], [], [], [], [], [], [], [], [],
    for n in range(n_dislocations):  # may also be zero
        random_idx_slab = np.random.randint(low=0, high=depth_list.shape[0])
        epi_lat = depth_list[random_idx_slab, 1]
        epi_lon = depth_list[random_idx_slab, 0]
        hypo_depth = - depth_list[random_idx_slab, 2]  # opposite sign for positive depths (Okada, 1985)
        depth_variability = -10 + 20 * params['uniform_vector'][i * n_dislocations + n, 0]
        if hypo_depth > 14.6:
            hypo_depth = hypo_depth + depth_variability
        if hypo_depth < 0:
            raise Exception('Negative depth')
        strike = strike_list[random_idx_slab, 2]
        dip = dip_list[random_idx_slab, 2]
        if 'rake_range' in params:
            min_rake, max_rake = params['rake_range']
            rake = min_rake + (max_rake - min_rake) * params['uniform_vector'][i * n_dislocations + n, 1]
        else:  # kept for compatibility
            rake = 75 + 25 * params['uniform_vector'][i, 1]  # rake from 75 to 100 deg
        min_mw, max_mw = 5, 7
        if 'magnitude_range' in params:
            min_mw, max_mw = params['magnitude_range']
        Mw = min_mw + (max_mw - min_mw) * params['uniform_vector'][i * n_dislocations + n, 2]
        Mo = 10 ** (1.5 * Mw + 9.1)
        stress_drop = params['lognormal_vector'][i * 3 + n]
        R = (7 / 16 * Mo / stress_drop) ** (1 / 3)
        u = 16 / (7 * np.pi) * stress_drop / 30e09 * R * 10 ** 3  # converted in mm in order to have displacement in mm
        L = np.sqrt(2 * np.pi) * R  # meters
        W = L * params['aspect_ratio']
        L = L * 10 ** (-3)  # conversion in km and then in lat, lon (suppose 1 degree ~ 100 km) for okada
        W = W * 10 ** (-3)
        if params['correct_latlon']:
            conv_coords = compute_geodesic_km_conversion_array(station_coordinates[:, :2], (epi_lat, epi_lon))
            lon_km, lat_km = conv_coords[:, 0], conv_coords[:, 1]
            displacement = okada85(lon_km, lat_km, 0, 0,
                                   hypo_depth + W / 2 * np.sin(np.deg2rad(dip)), L, W, u, 0, strike, dip, rake)
        else:
            displacement = okada85((station_coordinates[:, 1] - epi_lon) * 111.3194,
                                   (station_coordinates[:, 0] - epi_lat) * 111.3194, 0, 0,
                                   hypo_depth + W / 2 * np.sin(np.deg2rad(dip)), L, W, u, 0, strike, dip, rake)
        displacement_all.append(displacement)
        epi_lat_all.append(epi_lat)
        epi_lon_all.append(epi_lon)
        hypo_depth_all.append(hypo_depth)
        Mw_all.append(Mw)
        strike_all.append(strike)
        dip_all.append(dip)
        rake_all.append(rake)
        u_all.append(u)
        # stress_drop_all.append(stress_drop)
    return displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, Mw_all, strike_all, dip_all, rake_all, u_all  # , stress_drop_all


def synthetic_displacements_stations_cascadia_v2(n, station_coordinates, **kwargs):
    """Same as synthetic_displacements_stations_cascadia (SSEdetector), but more dislocations (or none) are computed
    for each sample. Also, we modify the stress drop to have a variation factor of 1.5. We return the displacement and
    the catalogue as a list, since they contain more dislocations."""
    if 'max_depth' in kwargs:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2(max_depth=kwargs['max_depth'])
    elif 'depth_range' in kwargs:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2(depth_range=kwargs['depth_range'])
    else:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2()
    uniform_vector = np.random.uniform(0, 1, (n * kwargs['max_n_disloc'], 3))  # account for multiple dislocations
    if kwargs['new_stress_drop']:
        mean_stress_drop = 10e3  # 10 KPa (Shearer et al. 2006)
        variation_factor = 1.5  # 10 # -> corresponding to a stress drop std of 15 KPa
    else:
        max_stress_drop = 0.1 * 1e06  # Gao et al., 2012
        min_stress_drop = 0.01 * 1e06  # Gao et al., 2012
        mean_stress_drop = 0.5 * (max_stress_drop + min_stress_drop)
        variation_factor = 10
    std_underlying_normal = np.sqrt(np.log(variation_factor ** 2 + 1))  # from coefficient of variation
    mean_underlying_normal = np.log(mean_stress_drop) - std_underlying_normal ** 2 / 2
    lognormal_vector = np.random.lognormal(mean_underlying_normal, std_underlying_normal,
                                           (n * kwargs['max_n_disloc'],))  # account for multiple dislocations
    # n_stations = station_coordinates.shape[0]
    # disp_stations = np.zeros((n, n_stations, 3))
    # catalogue = np.zeros((n, 9))
    results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=True)(
        delayed(_synthetic_displacement_stations_cascadia_v2)(i, admissible_depth, admissible_strike, admissible_dip,
                                                              station_coordinates, uniform_vector=uniform_vector,
                                                              lognormal_vector=lognormal_vector, **kwargs) for i in
        range(n))

    '''for i in range(n):
        for direction in range(3):
            disp_stations[i, :, direction] = results[i][0][direction]
        catalogue[i, :] = results[i][1:]'''

    displacement_list = [results[i][0] for i in range(n)]
    catalogue_list = [results[i][1:] for i in range(n)]
    return displacement_list, catalogue_list


def _sigmoid(x, alpha, beta, x0):
    return alpha / (1 + np.exp(-beta * (x - x0)))


def sigmoidal_rise_time(t, regime_value, duration, center, tol=0.01):
    gamma = regime_value * tol
    beta = 2 / duration * np.log((regime_value - gamma) / gamma)
    return _sigmoid(t, regime_value, beta, center)


def synthetic_sses_v2(n_samples, window_length, station_codes, station_coordinates, **kwargs):
    """As 'synthetic_sses' (SSEdetector) but we allow for the center of the sigmoid to be wherever in the window,
    as well as further features (cf. 'synthetic_displacements_stations_cascadia_v2')."""
    min_days, max_days = 10, 30
    synthetic_displacement, catalogue = synthetic_displacements_stations_cascadia_v2(n_samples, station_coordinates,
                                                                                     **kwargs)
    transients = np.zeros((n_samples, len(station_codes), window_length, 2))
    random_durations = np.random.randint(low=min_days, high=max_days, size=n_samples * kwargs['max_n_disloc'])
    random_center_values = np.random.randint(low=0, high=window_length, size=n_samples * kwargs['max_n_disloc'])

    transient_time_array = np.linspace(0, window_length, window_length)
    for sample in range(n_samples):
        for station in range(len(station_codes)):
            for direction in range(2):
                n_disloc = len(catalogue[sample][0])  # index 0 is used but all of them are equivalent
                for disloc in range(n_disloc):  # synth disp is a tuple, not np.array -> indexed as [direction][station]
                    transient = sigmoidal_rise_time(transient_time_array,
                                                    synthetic_displacement[sample][disloc][direction][station],
                                                    random_durations[sample * kwargs['max_n_disloc'] + disloc],
                                                    random_center_values[sample * kwargs['max_n_disloc'] + disloc])
                    transients[sample, station, :, direction] += transient

    return transients, random_durations, synthetic_displacement, catalogue

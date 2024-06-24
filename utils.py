import datetime
import io
import os
import re

import matplotlib
import numpy as np
import seaborn as sns
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import stats, interpolate
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, confusion_matrix


def _preliminary_operations(reference_period, detrend=True, **kwargs):
    """Returns GNSS data in reference_period. When detrending is used, the trends are not returned."""
    gnss_data, time_array = load_gnss_data_cascadia(reference_period=reference_period, **kwargs)
    if detrend:
        gnss_data = detrend_nan(time_array, gnss_data)  # gnss data is detrended first
    reference_time_indices = get_reference_period(time_array, reference_period)
    reference_time_array = time_array[reference_time_indices]
    selected_gnss_data = gnss_data[:, reference_time_indices, :]
    return selected_gnss_data, reference_time_array, gnss_data, time_array


def _decyr_time_array(reference_period):
    """Convention: the array starts on January 1st and ends on December 31."""
    lookup = ymd_decimal_year_lookup()
    start_date = datetime.date(year=reference_period[0], month=1, day=1)
    end_date = datetime.date(year=reference_period[1], month=12, day=31)
    dates_array = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    decyr_array = [lookup[date] for date in dates_array]
    decyr_array = np.array(decyr_array)
    return decyr_array


def _remove_stations(codes_to_remove, codes, coords, gnss_data):
    codes = np.array(codes)
    mask = ~np.isin(codes, codes_to_remove)
    new_coords = coords[mask]
    new_codes = codes[mask]
    new_gnss_data = gnss_data[mask]
    return new_codes, new_coords, new_gnss_data


def ymd_decimal_year_lookup():
    """Returns a lookup table for (year, month, day) to decimal year, with the convention introduced by Nasa JPL."""
    ymd_decimal_lookup = dict()
    with open('geo_data/decyr.txt', 'r') as f:
        next(f)
        for line in f:
            line = re.sub(' +', ' ', line)
            splitted_line = line.split(' ')
            decimal, year, month, day = splitted_line[1], splitted_line[2], splitted_line[3], splitted_line[4]
            decimal, year, month, day = float(decimal), int(year), int(month), int(day)
            ymd_decimal_lookup[(year, month, day)] = decimal
    return ymd_decimal_lookup


def load_gnss_data_cascadia(**kwargs):
    """Loads all the GNSS data from Cascadia and returns it along with a time array.
    The data is taken by considering the time span associated to the longest available time series.
    The unit measure is converted in mm."""
    n_directions = kwargs.pop('n_directions', 2)
    work_directory = os.path.expandvars(kwargs['work_directory']) if 'work_directory' in kwargs else './geo_data'
    station_codes, station_coordinates = cascadia_coordinates()
    n_stations = len(station_codes)
    file_lines = np.zeros((len(station_codes),))
    for i, code in enumerate(station_codes):  # check the longest file to have the largest time span
        num_lines = sum(1 for line in open(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{code}.txt')))
        file_lines[i] = num_lines
    target_station = station_codes[np.argmax(file_lines)]
    time_array = np.loadtxt(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{target_station}.txt'))[:, 0]
    # time_array = _decyr_time_array(kwargs['reference_period'])  # this will be used soon
    gnss_data = np.zeros((n_stations, len(time_array), n_directions))
    gnss_data.fill(np.nan)
    for i, code in enumerate(station_codes):
        data = np.loadtxt(os.path.join(work_directory, f'GNSS_CASCADIA/txt/{code}.txt'))[:, :n_directions + 1]
        correspondence_indices = np.searchsorted(time_array, data[:, 0])
        gnss_data[i, correspondence_indices, :] = data[:, 1:] * 1e03
    return gnss_data, time_array


def get_reference_period(time_array, period):
    """Returns indices in the data for the specified reference period, supposed to be a tuple of (start,end) dates."""
    reference_time_indices = np.where(np.logical_and(time_array > period[0], time_array < period[1]))[0]
    return reference_time_indices


def _first_and_last_seq(x, n):
    a = np.r_[n - 1, x, n - 1]
    a = a == n
    start = np.r_[False, ~a[:-1] & a[1:]]
    end = np.r_[a[:-1] & ~a[1:], False]
    return np.where(start | end)[0] - 1


def sequence_bounds(data, value):
    """Returns all the bounds for the sequences of {value} passed as input in the data. It does not handle
    1-sample sequences, which must be handled before."""
    return list(zip(*(iter(_first_and_last_seq(data, value)),) * 2))


def detrend_nan(x, data):
    detrended_data = np.zeros(data.shape)
    for observation in range(data.shape[0]):
        for direction in range(data.shape[2]):
            detrended_data[observation, :, direction] = _detrend_nan_1d(x, data[observation, :, direction])
    return detrended_data


def _detrend_nan_1d(x, y):
    # find linear regression line, subtract off data to detrend
    not_nan_ind = ~np.isnan(y)
    detrend_y = np.zeros(y.shape)
    detrend_y.fill(np.nan)
    if y[not_nan_ind].size > 0:
        m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
        detrend_y = y - (m * x + b)
    return detrend_y


def _detrend_nan_1d_v2(x, y):
    # find linear regression line, subtract off data to detrend
    not_nan_ind = ~np.isnan(y)
    detrend_y = np.zeros(y.shape)
    detrend_y.fill(np.nan)
    m, b = 0, 0
    if y[not_nan_ind].size > 0:
        m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
        detrend_y = y - (m * x + b)
    return detrend_y, m, b


def detrend_nan_v2(x, data):
    detrended_data = np.zeros(data.shape)
    trend_params = np.zeros((data.shape[0], data.shape[2], 2))
    for observation in range(data.shape[0]):
        for direction in range(data.shape[2]):
            detrended_, m, b = _detrend_nan_1d_v2(x, data[observation, :, direction])
            detrended_data[observation, :, direction] = detrended_
            trend_params[observation, direction, :] = [m, b]
    return detrended_data, trend_params


def cascadia_coordinates():
    """3D position of GNSS stations in Cascadia"""
    with open('geo_data/NGL_stations_cascadia.txt') as f:
        rows = f.read().splitlines()
        station_codes = []
        station_coordinates = []
        for row in rows:
            station_code, station_lat, station_lon, station_height = row.split(' ')
            station_codes.append(station_code)
            station_coordinates.append([station_lat, station_lon, station_height])
        station_coordinates = np.array(station_coordinates, dtype=np.float_)
    return station_codes, station_coordinates


def _find_nearest_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def plot_roc_curve(y_test, y_proba):
    """Utility function for Tensorboard. Plots the ROC curve and returns the figure."""
    figure = plt.figure(figsize=(8, 8))
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_proba)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.scatter(fpr_keras[_find_nearest_val(thresholds_keras, 0.5)[1]],
                tpr_keras[_find_nearest_val(thresholds_keras, 0.5)[1]])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    return figure


def plot_confusion_matrix(y_test, y_pred):
    """Utility function for Tensorboard. Plots the confusion matrix and returns the figure. The variable
    y_pred is assumed as an array obtained from y_proba after selecting a threshold."""
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d',
                xticklabels=['noise', 'noise+signal'], yticklabels=['noise', 'noise+signal'])  # fmt='g'
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def recall_as_function_of(y_true, y_pred, parameter, N_BINS=20):
    """
    author: Giuseppe Costantino
    Plots the binned recall as a function of the parameter passed as input.

    :param y_true: ground truth
    :param y_pred: predicted labels
    :param parameter: parameter with respect to which to plot the true positive ratio
    :param N_BINS: number of true positive ratio bins (default 20)
    :return: parameter bin positions, true positive ratio in a given parameter bin
    """
    true_pos_idx = np.where(np.logical_and(y_true == 1., y_pred == 1.))[0]
    false_neg_idx = np.where(np.logical_and(y_true == 1., y_pred == 0.))[0]
    # true positives and false negatives as function of the parameter
    param_true_pos = parameter[true_pos_idx]
    param_false_neg = parameter[false_neg_idx]
    _, bin_edges = np.histogram(parameter[np.where(y_true == 1.)[0]], bins=N_BINS)
    param_true_pos_bin_mean = []
    percentage_true_pos_bin_param = []
    for i in range(len(bin_edges) - 1):
        idx_bin_true_pos = np.where(np.logical_and(param_true_pos >= bin_edges[i], param_true_pos < bin_edges[i + 1]))[
            0]
        idx_bin_false_neg = \
            np.where(np.logical_and(param_false_neg >= bin_edges[i], param_false_neg < bin_edges[i + 1]))[0]
        if len(idx_bin_false_neg) > 0:
            percentage_true_pos_bin_param.append(
                len(idx_bin_true_pos) / (len(idx_bin_true_pos) + len(idx_bin_false_neg)))
        else:
            percentage_true_pos_bin_param.append(np.nan)
        param_true_pos_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    return param_true_pos_bin_mean, percentage_true_pos_bin_param


def n_true_pos_as_function_of(y_true, y_pred, parameter, N_BINS=20):
    """
    author: Giuseppe Costantino
    Plots the binned number of true positives as a function of the parameter passed as input.

    :param y_true: ground truth
    :param y_pred: predicted labels
    :param parameter: parameter with respect to which to plot the # true positives
    :param N_BINS: number of true positive ratio bins (default 20)
    :return: parameter bin positions, # true positives in a given parameter bin
    """
    true_pos_idx = np.where(np.logical_and(y_true == 1., y_pred == 1.))[0]
    false_neg_idx = np.where(np.logical_and(y_true == 1., y_pred == 0.))[0]
    # true positives and false negatives as function of the parameter
    param_true_pos = parameter[true_pos_idx]
    param_false_neg = parameter[false_neg_idx]
    _, bin_edges = np.histogram(parameter[np.where(y_true == 1.)[0]], bins=N_BINS)
    param_true_pos_bin_mean = []
    n_true_pos_bin_param = []
    for i in range(len(bin_edges) - 1):
        idx_bin_true_pos = np.where(np.logical_and(param_true_pos >= bin_edges[i], param_true_pos < bin_edges[i + 1]))[
            0]
        n_true_pos_bin_param.append(len(idx_bin_true_pos))
        param_true_pos_bin_mean.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    return param_true_pos_bin_mean, n_true_pos_bin_param


def plot_recall_function_of(y_test, y_pred, parameter, xlabel):
    mw_true_pos_bin_mean, percentage_true_pos_bin_mw = recall_as_function_of(y_test, y_pred, parameter)
    figure = plt.figure(figsize=(8, 8))
    plt.plot(mw_true_pos_bin_mean, percentage_true_pos_bin_mw)
    # plt.ylabel('% true positives (# true pos in a bin / #true pos)')
    plt.ylabel('True positive ratio (# true positives / positives)')
    plt.xlabel(xlabel)
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    import tensorflow as tf
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def get_cascadia_box(extended=False):
    # cascadia_box = [40 - 1, 51.8 - 0.2, -128.3 - 1, -121 + 0.5]  # min/max_latitude, min/max_longitude
    cascadia_box = [40 - 1, 51.8, -128.3, -121]  # min/max_latitude, min/max_longitude
    if extended:
        cascadia_box = [40 - 1, 51.8, -128.3, -119]
    return cascadia_box


def read_from_slab2(max_depth=60, depth_range=None):
    """Reads the slab2 grid and returns depth, strike, dip, and positions of points on the slab
    such that the depth is less than 60 km."""
    longitude_correction = [-360, 0, 0]
    depth_grid = np.loadtxt('geo_data/slab2/cas_slab2_dep_02.24.18.xyz', delimiter=',') + longitude_correction
    dip_grid = np.loadtxt('geo_data/slab2/cas_slab2_dip_02.24.18.xyz', delimiter=',') + longitude_correction
    strike_grid = np.loadtxt('geo_data/slab2/cas_slab2_str_02.24.18.xyz', delimiter=',') + longitude_correction

    if depth_range is None:
        ind_depth = np.where(depth_grid[:, 2] > - max_depth)[0]
    else:
        # 20 < depth < 40 --> -40 < depth < -20
        ind_depth = np.where(np.logical_and(depth_grid[:, 2] > -depth_range[1], depth_grid[:, 2] < -depth_range[0]))[0]

    region = [np.min(depth_grid[:, 1][ind_depth]), np.max(depth_grid[:, 1][ind_depth]),
              np.min(depth_grid[:, 0][ind_depth]),
              np.max(depth_grid[:, 0][ind_depth])]  # min/max_latitude, min/max_longitude
    # 13322 -> deep
    # 13315 -> shallow
    # 110904 -> nan

    '''cascadia_box = [40 - 1, 51.8 - 0.2, -128.3 - 1, -121 + 0.5]  # min/max_latitude, min/max_longitude
    cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                           urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                           lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                           resolution='i', projection='tmerc')
    # cascadia_map.drawmapboundary(fill_color='aqua')
    # cascadia_map.fillcontinents(color='grey', lake_color='aqua')
    cascadia_map.drawcoastlines()

    cascadia_map.drawparallels(np.arange(cascadia_box[0], cascadia_box[1], (cascadia_box[1] - cascadia_box[0]) / 4),
                               labels=[0, 1, 0, 0])
    cascadia_map.drawmeridians(np.arange(cascadia_box[2], cascadia_box[3], (cascadia_box[3] - cascadia_box[2]) / 4),
                               labels=[0, 0, 0, 1])
    # x, y = cascadia_map(depth_grid[:, 0][ind_depth], depth_grid[:, 1][ind_depth])
    # x, y = cascadia_map(dip_grid[:, 0][ind_depth], dip_grid[:, 1][ind_depth])
    x, y = cascadia_map(strike_grid[:, 0][ind_depth], strike_grid[:, 1][ind_depth])
    # plt.scatter(x, y, c=depth_grid[:, 2][ind_depth])
    # plt.scatter(x, y, c=dip_grid[:, 2][ind_depth])
    plt.scatter(x, y, c=strike_grid[:, 2][ind_depth])
    cmap = plt.colorbar()
    # cmap.set_label('Depth [km]')
    # cmap.set_label('Dip [°]')
    cmap.set_label('Strike [°]')
    plt.tight_layout()
    plt.show()'''
    return depth_grid[ind_depth], strike_grid[ind_depth], dip_grid[ind_depth], region


def cascadia_filtered_stations(n_selected_stations, reference_period=(2007, 2023), **kwargs):
    """Removes meaningless stations and returns the corresponding indices 'stations_subset_full',
    referring to the whole cascadia network."""
    gnss_data, _, _, _ = _preliminary_operations(reference_period, detrend=False, **kwargs)
    full_station_codes, full_station_coordinates = cascadia_coordinates()
    stations_to_remove = ['WSLB', 'YBHB', 'P687', 'BELI', 'PMAR', 'TGUA', 'OYLR', 'FTS5', 'RPT5', 'RPT6', 'P791',
                          'P674', 'P656', 'TWRI', 'WIFR', 'FRID', 'PNHG', 'COUR', 'SKMA', 'CSHR', 'HGP1', 'CBLV',
                          'PNHR', 'NCS2', 'TSEP', 'BCSC', 'LNG2']
    station_codes, station_coordinates, selected_gnss_data = _remove_stations(stations_to_remove, full_station_codes,
                                                                              full_station_coordinates, gnss_data)
    original_nan_pattern = np.isnan(selected_gnss_data[:, :, 0])
    n_nans_stations = original_nan_pattern.sum(axis=1)
    stations_subset = np.sort(np.argsort(n_nans_stations)[:n_selected_stations])
    station_codes_subset, station_coordinates_subset = np.array(station_codes)[stations_subset], station_coordinates[
                                                                                                 stations_subset, :]
    stations_subset_full = np.nonzero(np.in1d(full_station_codes, station_codes_subset))[0]
    return station_codes_subset, station_coordinates_subset, station_codes, station_coordinates, stations_subset_full


def _train_contour(cat):
    import alphashape
    alpha_shape = alphashape.alphashape(cat[:, :2], alpha=1.9)
    hull_pts = alpha_shape.exterior.coords.xy
    x_hull, y_hull = np.array(hull_pts[1]), np.array(hull_pts[0])
    # valid_pts_hull = np.where(x_hull < -121.95)[0]  # remove knot
    # x_hull, y_hull = x_hull[valid_pts_hull], y_hull[valid_pts_hull]
    dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
    interp_d_cat = np.linspace(dist_along[0], dist_along[-1], 350)
    interp_x_cat, interp_y_cat = interpolate.splev(interp_d_cat, spline)
    return interp_x_cat, interp_y_cat


def grid(x, y, z, resX=100, resY=100, method='linear'):
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    non_nan_idx = ~np.isnan(z)
    x, y, z = np.array(x), np.array(y), np.array(z)
    if method == 'cubic':
        interp = CloughTocher2DInterpolator(list(zip(x[non_nan_idx], y[non_nan_idx])), z[non_nan_idx])
        Z = interp(X, Y)
    elif method == 'linear':
        Z = griddata((x, y), z, (X, Y), method=method)
    else:
        Z = np.nan
    return X, Y, Z


def tremor_catalogue(north=False):
    """Return tremors from PNSN + Ide (2012) tremor catalogue.
    The dates are in the JPL-style format."""
    tremors = []
    date_table = ymd_decimal_year_lookup()
    catalogue_names = ['tremor_catalogue', 'tremor_catalogue2']
    for name in catalogue_names:
        with open(f'geo_data/{name}.txt') as f:
            next(f)
            for line in f:
                splitted_line = line.split(',')
                latitude = float(splitted_line[0].replace(' ', ''))
                longitude = float(splitted_line[1].replace(' ', ''))
                depth = float(splitted_line[2].replace(' ', ''))
                time = splitted_line[3][1:].split(' ')[0]
                year, month, day = time.split('-')
                year, month, day = float(year), float(month), float(day)
                decimal_date = date_table[(year, month, day)]
                tremors.append([latitude, longitude, depth, decimal_date])
    with open(f'geo_data/tremor_catalogue_ide.txt') as f:
        for line in f:
            splitted_line = line.split(',')
            latitude = float(splitted_line[2].replace(' ', ''))
            longitude = float(splitted_line[3].replace(' ', ''))
            depth = float(splitted_line[4].replace(' ', ''))
            year, month, day = splitted_line[0].split('-')
            year, month, day = float(year), float(month), float(day)
            decimal_date = date_table[(year, month, day)]
            tremors.append([latitude, longitude, depth, decimal_date])
    tremor_array = np.array(tremors)
    if north:
        valid_tremor_events = np.where(tremor_array[:, 0] >= 47.)[0]
        tremor_array = tremor_array[valid_tremor_events]
    return tremor_array


def sse_catalogue(return_magnitude=False, north=False):
    """Returns catalogued slow slip events from Michel et al., 2019, having the following format: start_date, end_date.
    Magnitude is returned too, according to the specified keyword parameter."""
    if return_magnitude:
        cols_to_use = (2, 3, 5)
    else:
        cols_to_use = (2, 3)
    filename = 'geo_data/catalogue_sse_michel_cascadia_mw.txt'
    catalog = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=cols_to_use)
    if north:
        valid_michel_events_north = [15, 18, 20, 22, 27, 33, 35, 37]
        catalog = catalog[valid_michel_events_north]
    return catalog


def get_n_tremors_per_day(time_span, tremor_catalogue):
    cascadia_box = [40 - 1, 51.8 - 0.2, -128.3 - 1, -121 + 0.5]  # min/max_latitude, min/max_longitude
    lat_cut = \
        np.where(np.logical_and(tremor_catalogue[:, 0] > cascadia_box[0], tremor_catalogue[:, 0] < cascadia_box[1]))[0]
    # time_cut = np.where(np.logical_and(tremor_catalogue[:, 3][lat_cut] > time_span[0], tremor_catalogue[:, 3][lat_cut] < time_span[-1]))[0]
    time_cut = np.where(np.logical_and(tremor_catalogue[:, 3][lat_cut] >= time_span[0],
                                       tremor_catalogue[:, 3][lat_cut] <= time_span[-1]))[0]
    tremors = tremor_catalogue[time_cut]
    unique, counts = np.unique(tremors[:, 3], return_counts=True)
    extended_counts = np.zeros(time_span.shape)
    for i in range(unique.shape[0]):
        idx_time = np.argwhere(time_span == unique[i])
        extended_counts[idx_time] = counts[i]
    return extended_counts


def overlap_percentage(xlist, ylist):
    min1 = min(xlist)
    max1 = max(xlist)
    min2 = min(ylist)
    max2 = max(ylist)

    overlap = max(0, min(max1, max2) - max(min1, min2))
    length = max1 - min1 + max2 - min2
    lengthx = max1 - min1
    lengthy = max2 - min2
    return 2 * overlap / length * 100, overlap / lengthx * 100, overlap / lengthy * 100


def plot_actual_vs_predicted_cascadia(y_test, y_pred, cat, density=True, filter=True):
    figure, axis = plt.subplots(1, 1, figsize=(14.4, 7.8))
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    axis.plot([p1, p2], [p1, p2], 'b-')
    if density:
        xy = np.vstack([y_test.ravel(), y_pred.ravel()])
        z = gaussian_kde(xy)(xy)
    else:
        z = y_test
    if filter:
        depth_filter = np.logical_and(cat[:, 2] > 20., cat[:, 2] < 40.)
        # y_test = y_test[mw_filter][depth_filter] ; y_pred = y_pred[mw_filter][depth_filter]
        _ = axis.scatter(y_test[~depth_filter], y_pred[~depth_filter], c=z[~depth_filter], alpha=0.3)
        _ = axis.scatter(y_test[depth_filter], y_pred[depth_filter], c=z[depth_filter], alpha=1.)
    else:
        sc = axis.scatter(y_test, y_pred, c=z)
    axis.set_xlabel('True Values')
    axis.set_ylabel('Predictions')
    if not density:
        cbar = figure.colorbar(sc)
        cbar.ax.set_ylabel('Actual label', rotation=270, labelpad=15)
    plt.title('Actual vs predicted')
    return figure


def plot_actual_vs_predicted_cascadia_multiple(y_test, y_pred, cat, density=True, filter=True, mw_axis=-1):
    n_var = y_test.shape[1]
    figure, axes = plt.subplots(1, n_var, figsize=(14.4, 7.8))
    for n in range(n_var):
        p1 = max(max(y_pred[:, n]), max(y_test[:, n]))
        p2 = min(min(y_pred[:, n]), min(y_test[:, n]))
        axes[n].plot([p1, p2], [p1, p2], 'b-')

        z = y_test[:, n_var + mw_axis]

        if density:
            xy = np.vstack([y_test[:, n], y_pred[:, n]])
            try:
                z = gaussian_kde(xy)(xy)
            except:
                pass
        if filter:
            depth_filter = np.logical_and(cat[:, 2] > 20., cat[:, 2] < 40.)
            # y_test = y_test[mw_filter][depth_filter] ; y_pred = y_pred[mw_filter][depth_filter]
            _ = axes[n].scatter(y_test[~depth_filter, n], y_pred[~depth_filter, n], c=z[~depth_filter], alpha=0.3)
            _ = axes[n].scatter(y_test[depth_filter, n], y_pred[depth_filter, n], c=z[depth_filter], alpha=1.)
        else:
            sc = axes[n].scatter(y_test[:, n], y_pred[:, n], c=z)
        axes[n].set_xlabel('True Values')
        axes[n].set_ylabel('Predictions')
    if not density:
        cbar = figure.colorbar(sc)
        cbar.ax.set_ylabel('Actual label', rotation=270, labelpad=15)
    return figure


def _draw_disp_field(axis, actual_disp, pred_disp, station_coordinates):
    displacements = [actual_disp, pred_disp]
    cascadia_box = [40 - 1.5, 51.8 - 0.2, -128.3 - 2.5, -121 + 2]  # min/max_latitude, min/max_longitude
    cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                           urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                           lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                           resolution='i', projection='lcc', ax=axis)  # , ax=ax9)

    cascadia_map.drawcoastlines(linewidth=0.5)
    cascadia_map.fillcontinents(color='bisque', lake_color='lightcyan')  # burlywood
    cascadia_map.drawmapboundary(fill_color='lightcyan')
    cascadia_map.drawmapscale(-122.5, 51.05, -122.5, 51.05, 300, barstyle='fancy', zorder=10)
    cascadia_map.drawparallels(np.arange(cascadia_box[0], cascadia_box[1], (cascadia_box[1] - cascadia_box[0]) / 4),
                               labels=[0, 1, 0, 0], linewidth=0.1)
    cascadia_map.drawmeridians(np.arange(cascadia_box[2], cascadia_box[3], (cascadia_box[3] - cascadia_box[2]) / 4),
                               labels=[0, 0, 0, 1], linewidth=0.1)
    cascadia_map.readshapefile('geo_data/tectonicplates-master/PB2002_boundaries', '', linewidth=0.3)
    colors = ['black', 'red']
    for i, disp in enumerate(displacements):
        max_length = np.max(np.sqrt(disp[:, 0] ** 2 + disp[:, 1] ** 2))
        # concatenate unit arrow
        unit_arrow_x, unit_arrow_y = -127., 39.5
        x = np.concatenate((station_coordinates[:, 1], [unit_arrow_x]))
        y = np.concatenate((station_coordinates[:, 0], [unit_arrow_y]))
        disp_x = np.concatenate((disp[:, 0], [-max_length]))
        disp_y = np.concatenate((disp[:, 1], [0.]))

        cascadia_map.quiver(x, y, disp_x, disp_y, color=colors[i], latlon=True, width=0.0035)

    axis.legend(['actual', 'predicted'])


def plot_disp_field_cascadia(y_test, y_pred, cat, station_coordinates):
    mw_6_idx = np.where(cat[:, 3] < 6.1)[0][0]
    mw_6_5_idx = 0
    mw_idx = [mw_6_idx, mw_6_5_idx]
    titles = ['Mw 6', 'Mw 6.5']
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.8))
    for i, axis in enumerate(axes):
        _draw_disp_field(axis, y_test[mw_idx[i]], y_pred[mw_idx[i]], station_coordinates)
        axis.set_title(titles[i])

    return figure


def _full_time_series_denoising_plots(X_test_loader, y_test, y_pred, cat, station_coordinates, direction=0):
    '''no_disloc_idx = 6  # 0
    one_disloc_idx = 11  # 6
    two_disloc_idx = 9  # 11
    three_disloc_idx = 15  # 3'''
    no_disloc_idx, one_disloc_idx, two_disloc_idx, three_disloc_idx = None, None, None, None
    for i in range(len(cat)):
        if len(cat[i][0]) == 0:
            no_disloc_idx = i

        if len(cat[i][0]) == 1:
            one_disloc_idx = i

        if len(cat[i][0]) == 2:
            two_disloc_idx = i

        if len(cat[i][0]) == 3:
            three_disloc_idx = i
        if no_disloc_idx is not None and one_disloc_idx is not None and two_disloc_idx is not None and three_disloc_idx is not None:
            break

    for i in range(100):  # modify after
        if len(cat[i][0]) == 10:
            one_disloc_idx = i

        if len(cat[i][0]) == 20:
            two_disloc_idx = i

        if len(cat[i][0]) == 30:
            three_disloc_idx = i

    for i, d in enumerate(X_test_loader):
        X_test = d[0]
        break

    latsort = np.argsort(station_coordinates[:, 0])

    figure, axis = plt.subplots(4, 3, figsize=(11, 14))
    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    vmin, vmax = -5, 10
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    cmap.set_bad('black', 1.)

    norm_other = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

    m = axis[0][0].matshow(X_test[no_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm)
    plt.colorbar(m, ax=axis[0][0])
    m = axis[0][1].matshow(y_test[no_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[0][1])
    m = axis[0][2].matshow(y_pred[no_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[0][2])
    axis[0][2].title.set_text(f'No dislocation')

    m = axis[1][0].matshow(X_test[one_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm)
    plt.colorbar(m, ax=axis[1][0])
    m = axis[1][1].matshow(y_test[one_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[1][1])
    m = axis[1][2].matshow(y_pred[one_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[1][2])
    axis[1][2].title.set_text(f'Mw {cat[one_disloc_idx][3][0]:.2f}')

    m = axis[2][0].matshow(X_test[two_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm)
    plt.colorbar(m, ax=axis[2][0])
    m = axis[2][1].matshow(y_test[two_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[2][1])
    m = axis[2][2].matshow(y_pred[two_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[2][2])
    axis[2][2].title.set_text(f'Mw {cat[two_disloc_idx][3][0]:.2f}, {cat[two_disloc_idx][3][1]:.2f}')

    m = axis[3][0].matshow(X_test[three_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm)
    plt.colorbar(m, ax=axis[3][0])
    m = axis[3][1].matshow(y_test[three_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[3][1])
    m = axis[3][2].matshow(y_pred[three_disloc_idx, latsort, :, direction], cmap=cmap, norm=norm_other)
    plt.colorbar(m, ax=axis[3][2])
    axis[3][2].title.set_text(
        f'Mw {cat[three_disloc_idx][3][0]:.2f}, {cat[three_disloc_idx][3][1]:.2f}, {cat[three_disloc_idx][3][2]:.2f}')

    plt.tight_layout()

    return figure


def _static_denoising_plots(y_test, y_pred, cat, station_coordinates):
    no_disloc_idx = 6  # 0
    one_disloc_idx = 11  # 6
    two_disloc_idx = 9  # 11
    three_disloc_idx = 15  # 3

    figure, axes = plt.subplots(2, 2, figsize=(11, 14))

    _draw_disp_field(axes[0][0], y_test[no_disloc_idx], y_pred[no_disloc_idx], station_coordinates)
    axes[0][0].title.set_text(f'No dislocation')
    _draw_disp_field(axes[0][1], y_test[one_disloc_idx], y_pred[one_disloc_idx], station_coordinates)
    axes[0][1].title.set_text(f'One disl. Mw {cat[one_disloc_idx][3][0]:.2f}')
    _draw_disp_field(axes[1][0], y_test[two_disloc_idx], y_pred[two_disloc_idx], station_coordinates)
    axes[1][0].title.set_text(f'Two disl. Mw {cat[two_disloc_idx][3][0]:.2f}, {cat[two_disloc_idx][3][1]:.2f}')
    _draw_disp_field(axes[1][1], y_test[three_disloc_idx], y_pred[three_disloc_idx], station_coordinates)
    axes[1][1].title.set_text(
        f'Three disl. Mw {cat[three_disloc_idx][3][0]:.2f}, {cat[three_disloc_idx][3][1]:.2f}, {cat[three_disloc_idx][3][2]:.2f}')

    return figure


def denoising_plots(X_test_loader, y_test, y_pred, cat, station_coordinates, direction=0, static=False):
    if static:
        figure = _static_denoising_plots(y_test, y_pred, cat, station_coordinates)
    else:
        figure = _full_time_series_denoising_plots(X_test_loader, y_test, y_pred, cat, station_coordinates, direction)
    return figure


def _geodesic_km_conversion(lat1, lon1, lat2, lon2):
    latdist = np.sign(lat1 - lat2) * geodesic((lat1, lon1), (lat2, lon1)).km
    londist = np.sign(lon1 - lon2) * geodesic((lat1, lon1), (lat1, lon2)).km
    return londist, latdist


def compute_geodesic_km_conversion_array(coords, point):
    conv_coords = np.zeros(coords.shape)
    for i in range(coords.shape[0]):
        conv_coords[i] = _geodesic_km_conversion(coords[i, 0], coords[i, 1], point[0], point[1])
    return conv_coords

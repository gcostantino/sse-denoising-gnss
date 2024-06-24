import datetime
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from sklearn.cluster import DBSCAN

from utils import _preliminary_operations, cascadia_filtered_stations, tremor_catalogue, \
    ymd_decimal_year_lookup, _find_nearest_val


def generate_denoised_time_series():
    reference_period = (2007, 2023)
    window_length = 60
    n_selected_stations = 200

    with np.load('predictions/pred_SSEdenoiser.npz') as f:
        full_denoised_real_data = f['pred']

    full_denoised_real_data = np.gradient(full_denoised_real_data, axis=2)

    selected_gnss_data, selected_time_array, _, _ = _preliminary_operations(reference_period, detrend=False)

    # we take a sliding window and we average the displacement over time
    offset_cut = 20
    averaged_continuous_denoised_data = np.zeros(
        (selected_time_array.shape[0] - window_length - 2 * offset_cut, n_selected_stations, 2))

    # for i in range(full_denoised_real_data.shape[0]):
    for i in range(averaged_continuous_denoised_data.shape[0] - offset_cut):
        averaged_continuous_denoised_data[i:i + (window_length - 2 * offset_cut)] += np.transpose(
            full_denoised_real_data[i, :, offset_cut:window_length - offset_cut, :], (1, 0, 2))

    num_windows_per_sample = np.zeros((len(averaged_continuous_denoised_data)))
    num_windows_per_sample[:(window_length - 2 * offset_cut)] = np.arange(1, (window_length - 2 * offset_cut + 1))
    num_windows_per_sample[-(window_length - 2 * offset_cut):] = np.arange(1, (window_length - 2 * offset_cut + 1))[
                                                                 ::-1]
    num_windows_per_sample[
    (window_length - 2 * offset_cut):-(window_length - 2 * offset_cut)] = window_length - 2 * offset_cut

    averaged_continuous_denoised_data = averaged_continuous_denoised_data / num_windows_per_sample[
        ..., np.newaxis, np.newaxis]
    return selected_time_array, averaged_continuous_denoised_data


def _latitude_time_plot(time, data, tremors, station_coordinates, latsort, tol=0.01, window_length=60, offset=0,
                        static=False, downsample_tremors=False, draw_tremors=True, tremor_alpha=1.,
                        data_pcolormesh=False, fig_path='denoising_figures/overall_lat_time.pdf', zoom=False):
    data[np.abs(data) < tol] = np.nan
    data[data > 0] = np.nan
    vmin, vmax = -0.1, 0
    # vmin, vmax = -4, 0
    # vmin, vmax = np.nanmin(data), np.nanmax(data)
    cmap = matplotlib.cm.get_cmap("turbo_r").copy()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # figure = plt.figure(figsize=(16, 8), dpi=300)
    figure = plt.figure(figsize=(16, 8), dpi=100)
    if static:
        x, y = np.meshgrid(time[window_length // 2:-window_length // 2], station_coordinates[latsort, 0])
    else:
        if zoom:
            x, y = np.meshgrid(time, station_coordinates[latsort, 0])
        else:
            x, y = np.meshgrid(time[0 + offset:-window_length - offset], station_coordinates[latsort, 0])
    if data_pcolormesh:
        f = interp2d(x, y, data[:, latsort, 0].T, kind='cubic')
        x_up = np.linspace(time[window_length // 2], time[-window_length // 2], len(x) * 1)
        y_up = np.linspace(station_coordinates[latsort, 0][0], station_coordinates[latsort, 0][-1], len(y) * 1)
        data1 = f(x_up, y_up)
        Xn, Yn = np.meshgrid(x_up, y_up)
        plt.pcolormesh(Xn, Yn, data1, cmap=cmap, norm=norm, zorder=0)
        # plt.pcolormesh(x, y, data[:, latsort, 0].T,  cmap=cmap, norm=norm, zorder=0, antialiased=True, shading='gouraud')
    else:
        # plot in reverse order to avoid to mask the sse growth
        plt.scatter(x[:, ::-1], y[:, ::-1], c=data[:, latsort, 0].T[:, ::-1], cmap=cmap, norm=norm, s=10, alpha=0.7,
                    zorder=0, edgecolors='none')
        # plt.scatter(x, y, c=data[:, latsort, 0].T, cmap=cmap, norm=norm, s=10, alpha=0.5, zorder=0, edgecolors='none')
        cbar = plt.colorbar()
        cbar.solids.set_alpha(1)
        cbar.ax.set_ylabel('Displacement rate [mm/day]', rotation=270, labelpad=25, size=13)

    if draw_tremors:
        tremor_scatter_size = 0.2
        if downsample_tremors:
            fraction_points_per_cluster = 0.05
            # we only downsample tremors for PNSN catalogue
            idx_pnsn = tremors[:, 3] > 2009
            # ide's catalogue is kept as it is
            plt.scatter(tremors[~idx_pnsn, 3], tremors[~idx_pnsn, 0], s=tremor_scatter_size, alpha=tremor_alpha,
                        color='black', zorder=1)
            dbscan = DBSCAN(eps=0.1, min_samples=10).fit(tremors[idx_pnsn][:, (0, 3)])
            labels = dbscan.labels_

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_points = np.where(labels == label)[0]
                n_points_per_cluster = int(fraction_points_per_cluster * len(cluster_points))
                # print('#points:', n_points_per_cluster)
                selected_indices = np.random.choice(cluster_points, size=n_points_per_cluster)
                plt.scatter(tremors[idx_pnsn][selected_indices, 3], tremors[idx_pnsn][selected_indices, 0],
                            s=tremor_scatter_size, alpha=tremor_alpha, color='black', zorder=1)
        else:
            plt.scatter(tremors[:, 3], tremors[:, 0], s=tremor_scatter_size, alpha=tremor_alpha, color='black',
                        zorder=1)
    plt.ylabel('Latitude')
    plt.xlabel('Time [years]')
    plt.show()
    '''plt.savefig(fig_path, bbox_inches='tight')
    plt.close(figure)'''


def compare_xue_freymuller(station_codes, station_coordinates, zoom=False):
    zoom_start, zoom_end = 2009.6, 2012
    path = 'data_xue_freymuller/ml_rawresults'

    if zoom:
        # time array from decimal year to datetime
        lookup = ymd_decimal_year_lookup()
        inv_lookup = {v: k for k, v in lookup.items()}

    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    xue_stations = []
    for filename in sorted(filenames):
        station_name = filename.split('.')[0]
        xue_stations.append(station_name)

    xue_data = []
    xue_valid_stations = []
    xue_valid_stations_idx = []
    # print(xue_stations)

    xue_time = np.loadtxt(path + '/albh.txt', delimiter=' ', usecols=0)
    xue_data = np.zeros((len(station_codes), len(xue_time), 3))
    # xue_data.fill(np.nan)

    for i, code in enumerate(station_codes.tolist()):
        if code.lower() in xue_stations:
            # print('loading:', path + '/' + code.lower() + '.txt')
            data = np.loadtxt(path + '/' + code.lower() + '.txt', delimiter=' ')
            if len(data) > 0:
                xue_valid_stations.append(code)
                xue_valid_stations_idx.append(i)
                correspondence_indices = np.searchsorted(xue_time, data[:, 0])
                xue_data[i, correspondence_indices, :] = data[:, (10, 9, 11)]

    print(xue_data.shape)
    print(xue_time.shape)

    with np.load('denoised_ts.npz') as f:
        time, data, tremors = f['time'], f['data'], f['tremors']

    def decyr_to_datetime(date, time_array):
        if date not in inv_lookup:
            date, _ = _find_nearest_val(time_array, date)  # find closest date

        datetime_date = datetime.datetime(inv_lookup[date][0], inv_lookup[date][1], inv_lookup[date][2])
        return datetime_date

    if zoom:
        # zoom over specific time period
        xue_zoom_idx = np.where((xue_time >= zoom_start) & (xue_time <= zoom_end))[0]
        xue_time = xue_time[xue_zoom_idx]
        xue_data = xue_data[:, xue_zoom_idx, :]

        xue_time = np.array([decyr_to_datetime(date, time) for date in xue_time])

    else:
        time_cut = np.logical_and(xue_time > 2007, xue_time < 2016)
        xue_data = xue_data[:, time_cut, :]
        xue_time = xue_time[time_cut]

    # xue_data[xue_data < 0.5] = np.nan

    prob_max = np.maximum(np.maximum(xue_data[..., 0], xue_data[..., 1]), xue_data[..., 2])
    # prob_max[prob_max < 0.5] = np.nan

    '''latsort = np.argsort(station_coordinates[xue_valid_stations_idx, 0])[::-1]
    plt.matshow(prob_max[xue_valid_stations_idx][latsort], aspect='auto')
    plt.colorbar()
    plt.show()'''

    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    epoch_mat, stalat_mat = np.meshgrid(xue_time, station_coordinates[:, 0])

    index_pos = prob_max > 0.5
    prob_vec_pos = prob_max[index_pos]
    epoch_vec_pos = epoch_mat[index_pos]
    stalat_vec_pos = stalat_mat[index_pos]

    index_sort = np.argsort(prob_vec_pos)
    prob_vec_pos = prob_vec_pos[index_sort]
    epoch_vec_pos = epoch_vec_pos[index_sort]
    stalat_vec_pos = stalat_vec_pos[index_sort]

    offset, window_length, tol = 20, 60, 0.01
    time = time[0 + offset:-window_length - offset]
    time_cut_ours = np.logical_and(time > 2007, time < 2016)
    time = time[time_cut_ours]
    data = data[time_cut_ours]
    tremors = tremors[np.logical_and(tremors[:, 3] > 2007, tremors[:, 3] < 2016)]
    tremors = tremors[np.logical_and(tremors[:, 0] > 39.8, tremors[:, 0] < 51)]

    data[np.abs(data) < tol] = np.nan
    data[data > 0] = np.nan
    print(data.shape)
    print(time.shape)

    if zoom:
        # zoom over specific time period
        our_zoom_idx = np.where((time >= zoom_start) & (time <= zoom_end))[0]
        idx_zoom_tremors = np.where((tremors[:, 3] >= zoom_start) & (tremors[:, 3] <= zoom_end))[0]

        time = time[our_zoom_idx]
        data = data[our_zoom_idx]

        tremors = tremors[idx_zoom_tremors]

        time = np.array([decyr_to_datetime(date, None) for date in time])
        tremor_time = np.array([decyr_to_datetime(date, None) for date in tremors[:, 3]])

    draw_tremors = True
    downsample_tremors = True
    x, y = np.meshgrid(time, station_coordinates[latsort, 0])
    vmin, vmax = -0.1, 0
    cmap = matplotlib.cm.get_cmap("turbo_r").copy()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues_r
    '''start_percentage = 0.5  # Adjust this value as needed
    truncated_blues_cmap = LinearSegmentedColormap.from_list(
        'Truncated Reds', cmap(np.linspace(start_percentage, 1, 256))
    )'''
    colors = cmap(np.linspace(-0.4, 1, 10))
    truncated_blues_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

    max_sort_disp = np.argsort(-data[:, latsort, 0].T.flatten())
    # max_sort_disp_xy = np.argsort(-data[:, latsort, 0].T)

    draw_xue_freymueller = True
    draw_costantino = True

    # fig = plt.figure(figsize=(10, 6), dpi=300)
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    xue_point_size = 30
    if draw_xue_freymueller:
        s1 = plt.scatter(epoch_vec_pos, stalat_vec_pos, s=xue_point_size, c=prob_vec_pos, cmap=plt.cm.Reds, alpha=0.2,
                         zorder=10, edgecolors='none')
    if draw_costantino:
        s2 = plt.scatter(x.flatten()[max_sort_disp], y.flatten()[max_sort_disp], s=10,
                         c=data[:, latsort, 0].T.flatten()[max_sort_disp], cmap=truncated_blues_cmap, norm=norm,
                         alpha=.2, zorder=9)

    if draw_tremors:
        tremor_scatter_size = 0.2
        if downsample_tremors:
            fraction_points_per_cluster = 0.05
            # we only downsample tremors for PNSN catalogue
            idx_pnsn = tremors[:, 3] > 2009.5
            # ide's catalogue is kept as it is
            if not zoom:
                plt.scatter(tremors[~idx_pnsn, 3], tremors[~idx_pnsn, 0], s=tremor_scatter_size, alpha=1.,
                            color='black',
                            zorder=12)
            else:
                plt.scatter(tremor_time[~idx_pnsn], tremors[~idx_pnsn, 0], s=tremor_scatter_size, alpha=1.,
                            color='black', zorder=12)
            dbscan = DBSCAN(eps=0.1, min_samples=10).fit(tremors[idx_pnsn][:, (0, 3)])
            labels = dbscan.labels_

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_points = np.where(labels == label)[0]
                n_points_per_cluster = int(fraction_points_per_cluster * len(cluster_points))
                # print('#points:', n_points_per_cluster)
                selected_indices = np.random.choice(cluster_points, size=n_points_per_cluster)
                if not zoom:
                    plt.scatter(tremors[idx_pnsn][selected_indices, 3], tremors[idx_pnsn][selected_indices, 0],
                                s=tremor_scatter_size, alpha=1., color='black', zorder=12)
                else:
                    plt.scatter(tremor_time[idx_pnsn][selected_indices], tremors[idx_pnsn][selected_indices, 0],
                                s=tremor_scatter_size, alpha=1., color='black', zorder=12)
        else:
            if not zoom:
                plt.scatter(tremors[:, 3], tremors[:, 0], s=tremor_scatter_size, alpha=1., color='black', zorder=12)
            else:
                plt.scatter(tremor_time, tremors[:, 0], s=tremor_scatter_size, alpha=1., color='black', zorder=12)

    plt.grid(True, 'both', 'both')
    plt.xlabel('Year')
    plt.ylabel('Latitude')

    pos = ax.get_position()
    bar_h = (pos.y1 - pos.y0) * 0.45  # 0.5 joins the two bars, e.g. 0.48 separates them a bit
    if draw_xue_freymueller:
        ax_cbar1 = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.01, bar_h])
        cbar = fig.colorbar(s1, cax=ax_cbar1, orientation='vertical')
        cbar.solids.set(alpha=1)
        cbar.ax.set_ylabel('Probability', rotation=270, labelpad=15)
    if draw_costantino:
        ax_cbar2 = fig.add_axes([pos.x1 + 0.02, pos.y1 - bar_h, 0.01, bar_h])
        cbar2 = fig.colorbar(s2, cax=ax_cbar2, orientation='vertical')
        cbar2.solids.set(alpha=1)
        cbar2.ax.set_ylabel('Displacement rate [mm/day]', rotation=270, labelpad=15)

    # cbar_depth = plt.colorbar(pcm_depth, orientation='horizontal', fraction=0.018, pad=0.065, shrink=0.6,
    #                                   ticks=[10, 20, 30, 40, 50])  # fraction=0.046, pad=0.04)
    #         cbar_depth.ax.set_xlabel('Slab depth (km)', labelpad=5)

    # cbar.set_ticks([0.5, 0.75, 1.0])

    # cbar.minorticks_on()
    # cbar.ax.set_title('Probability', fontsize=10)
    # cbar2.ax.set_title('Displacement rate [mm]', fontsize=10)

    # plt.clim(0.5, 1.0)
    # plt.xlim(2005, 2016)
    # plt.ylim(40, 51)
    # plt.xticks(np.arange(2005, 2017))
    # plt.yticks(np.arange(40, 52))
    # ax.set_yticks(np.arange(40, 51, 0.5), minor=True)
    # ax.set_xticks(np.arange(2005, 2016, 0.25), minor=True)

    zoom_str = '_zoom' if zoom else ''

    # plt.show()
    figure_title = 'tremors' + ('+freymueller' if draw_xue_freymueller else '') + (
        '+costantino' if draw_costantino else '')
    plt.savefig(f'{figure_title}_size{xue_point_size}{zoom_str}.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    tremors = tremor_catalogue()
    tremors = tremors[tremors[:, 0] > 39]

    time, denoised_ts = generate_denoised_time_series()
    np.savez('denoised_ts', time=time, data=denoised_ts, tremors=tremors)
    # run the previous two lines just once

    with np.load('denoised_ts.npz') as f:
        time, data, tremors = f['time'], f['data'], f['tremors']
    n_selected_stations = 200
    station_codes, station_coordinates, full_station_codes, full_station_coordinates, station_subset = cascadia_filtered_stations(
        n_selected_stations)
    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    window_length = 60
    tol = 0.02
    offset_cut = 20
    _latitude_time_plot(time, data, tremors, station_coordinates, latsort,
                        tol=tol, offset=offset_cut, window_length=window_length, downsample_tremors=True,
                        draw_tremors=True, data_pcolormesh=False, fig_path='denoising_figures/overall_lat_time.png')

    compare_xue_freymuller(station_codes, station_coordinates, zoom=True)

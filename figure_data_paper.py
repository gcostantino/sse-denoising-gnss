import multiprocessing
import os.path

import alphashape
import joblib
import matplotlib
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import interpolate
from sklearn.cluster import DBSCAN

from sse_generator_v2.okada import forward as okada85
from sse_generator_v2.synthetic_sse_cascadia import sigmoidal_rise_time
from utils import cascadia_filtered_stations, read_from_slab2, grid, tremor_catalogue

np.random.seed(0)

plt.rc('font', family='Helvetica')

SMALL_SIZE = 8 + 2 + 2  # + 6  # +6 only for disp
LEGEND_SIZE = 8 + 2 + 2  # + 6
MEDIUM_SIZE = 10 + 2 + 4  # + 6
BIGGER_SIZE = 12 + 2 + 2  # + 6

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def _train_contour_mod(cat, cascadia_map):
    import alphashape
    ext_lat, ext_lon = [], []
    for i in range(len(cat)):
        if cat[i][0] != []:
            ext_lat += cat[i][0]
            ext_lon += cat[i][1]
    ext_cat = np.vstack((ext_lat, ext_lon)).T
    # alpha_shape = alphashape.alphashape(ext_cat, alpha=1.9)
    # cascadia_map.scatter(ext_lon, ext_lat, latlon=True)
    # plt.show()
    ext_cat = ext_cat[:1000]
    alpha_shape = alphashape.alphashape(ext_cat, alpha=1.9)
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


def _load_synth_data(filename, load_pred_only=False):
    data_dict = joblib.load(filename)
    pred = data_dict['pred']
    if load_pred_only:
        return pred
    X = data_dict['X']
    y = data_dict['y']
    catalogue = data_dict['cat']
    return X, y, catalogue, pred


def _draw_table_data(data_window, time_span, station_coordinates, every_nth=17, title='', figtitle=None, vmin=None,
                     vmax=None, colorbar=False, ylabel=True, folder='fig_synth_data'):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    if vmin is None:
        vmin = -5
    if vmax is None:
        vmax = 10
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    cmap.set_bad('black', 1.)
    matshow_extent = [time_span[0], time_span[1], station_coordinates[:, 0].min(),
                      station_coordinates[:, 0].max()]
    mat = ax.matshow(data_window[latsort, :, 0], cmap=cmap, norm=norm, aspect='auto',  # , aspect='auto',
                     extent=matshow_extent)
    if ylabel:
        ax.set_ylabel('Latitude [°]', labelpad=7)
    ax.set_xlabel('Time [days]')
    ax.set_title(title)
    ax.set_yticks(np.around(station_coordinates[latsort, 0], decimals=1))
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
            ax.yaxis.get_major_ticks()[n].set_visible(False)
    if colorbar:
        cbar = fig.colorbar(mat, ax=ax)
        cbar.ax.set_ylabel('E-W displacement [mm]', rotation=270, labelpad=17)

    # plt.tight_layout()
    plt.savefig(f'{os.path.join(folder, figtitle)}.pdf', bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def _synthetic_displacement_stations_cascadia_v2(i, depth_list, strike_list, dip_list, station_coordinates, **params):
    """Same as '_synthetic_displacement_stations_cascadia'. What's new:
    - more dislocations can be associated to the same sample. We return them a a list."""
    n_dislocations = 3
    displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, Mw_all, strike_all, dip_all, rake_all, u_all, stress_drop_all = [], [], [], [], [], [], [], [], [], [],
    L_all, W_all = [], []
    for n in range(n_dislocations):  # may also be zero
        random_idx_slab = np.random.randint(low=0, high=depth_list.shape[0])
        epi_lat = depth_list[random_idx_slab, 1]
        epi_lon = depth_list[random_idx_slab, 0]
        hypo_depth = - depth_list[random_idx_slab, 2]  # opposite sign for positive depths (Okada, 1985)
        depth_variability = -10 + 20 * params['uniform_vector'][i * 3 + n, 0]
        if hypo_depth > 14.6:
            hypo_depth = hypo_depth + depth_variability
        if hypo_depth < 0:
            raise Exception('Negative depth')
        strike = strike_list[random_idx_slab, 2]
        dip = dip_list[random_idx_slab, 2]
        if 'rake_range' in params:
            min_rake, max_rake = params['rake_range']
            rake = min_rake + (max_rake - min_rake) * params['uniform_vector'][i * 3 + n, 1]
        else:  # kept for compatibility
            rake = 75 + 25 * params['uniform_vector'][i, 1]  # rake from 75 to 100 deg
        min_mw, max_mw = 5, 7
        if 'magnitude_range' in params:
            min_mw, max_mw = params['magnitude_range']
        Mw = min_mw + (max_mw - min_mw) * params['uniform_vector'][i * 3 + n, 2]
        Mo = 10 ** (1.5 * Mw + 9.1)
        stress_drop = params['lognormal_vector'][i * 3 + n]
        R = (7 / 16 * Mo / stress_drop) ** (1 / 3)
        u = 16 / (7 * np.pi) * stress_drop / 30e09 * R * 10 ** 3  # converted in mm in order to have displacement in mm
        L = np.sqrt(2 * np.pi) * R  # meters
        W = L / 2
        L = L * 10 ** (-3)  # conversion in km and then in lat, lon (suppose 1 degree ~ 100 km) for okada
        W = W * 10 ** (-3)
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
        stress_drop_all.append(stress_drop)
        L_all.append(L)
        W_all.append(W)
    # return displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, Mw_all, strike_all, dip_all, rake_all, u_all, stress_drop_all
    return displacement_all, epi_lat_all, epi_lon_all, hypo_depth_all, Mw_all, strike_all, dip_all, rake_all, L_all, W_all, u_all, stress_drop_all


def synthetic_displacements_stations_cascadia_v2_mod(n, station_coordinates, **kwargs):
    """Same as synthetic_displacements_stations_cascadia, but more dislocations (or none) are computed for each sample.
    Also, we modify the stress drop to have a variation factor of 1.5. We return the displacement and the catalogue
    as a list, since they contain more dislocations."""
    if 'max_depth' in kwargs:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2(max_depth=kwargs['max_depth'])
    elif 'depth_range' in kwargs:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2(depth_range=kwargs['depth_range'])
    else:
        admissible_depth, admissible_strike, admissible_dip, region = read_from_slab2()
    uniform_vector = np.random.uniform(0, 1, (n * 3, 3))  # account for multiple dislocations
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
                                           (n * 3,))  # account for multiple dislocations
    # n_stations = station_coordinates.shape[0]
    # disp_stations = np.zeros((n, n_stations, 3))
    # catalogue = np.zeros((n, 9))
    results = Parallel(n_jobs=1, verbose=True)(
        delayed(_synthetic_displacement_stations_cascadia_v2)(i, admissible_depth, admissible_strike, admissible_dip,
                                                              station_coordinates, uniform_vector=uniform_vector,
                                                              lognormal_vector=lognormal_vector, **kwargs) for i in
        range(n))

    displacement_list = [results[i][0] for i in range(n)]
    catalogue_list = [results[i][1:] for i in range(n)]
    return displacement_list, catalogue_list


def synthetic_sses_v2_mod(window_length, station_codes, station_coordinates, **kwargs):
    """As 'synthetic_sses' but we allow for the center of the sigmoid to be wherever in the window,
    as well as further features (cf. 'synthetic_displacements_stations_cascadia_v2'."""
    n_samples = 1
    synthetic_displacement, catalogue = synthetic_displacements_stations_cascadia_v2_mod(n_samples, station_coordinates,
                                                                                         **kwargs)
    transients = np.zeros((n_samples, len(station_codes), window_length, 2))
    random_durations = [20, 20, 40]
    random_center_values = [30, 40, 30]
    single_transients = np.zeros((n_samples, 3, len(station_codes), window_length, 2))

    transient_time_array = np.linspace(0, window_length, window_length)
    for sample in range(n_samples):
        for station in range(len(station_codes)):
            for direction in range(2):
                n_disloc = len(catalogue[sample][0])  # index 0 is used but all of them are equivalent
                for disloc in range(n_disloc):  # synth disp is a tuple, not np.array -> indexed as [direction][station]
                    transient = sigmoidal_rise_time(transient_time_array,
                                                    synthetic_displacement[sample][disloc][direction][station],
                                                    random_durations[sample * 3 + disloc],
                                                    random_center_values[sample * 3 + disloc])
                    transients[sample, station, :, direction] += transient
                    single_transients[sample, disloc, station, :, direction] = transient

    return transients, random_durations, synthetic_displacement, catalogue, single_transients


def map_figure_synth(synthetic_displacement, station_coordinates, dislocation_parameters, train_catalogue,
                     draw_tremors=True, draw_disp_field=True, draw_gnss_network=True, draw_dislocation=True,
                     draw_train_contour=True, draw_isodepth=True, fig_directory='fig_synth_data', color_code_depth=True,
                     update=False, cascadia_map=None, final_update=False, arrow_color=None, fig=None, id_synth=0):
    from shapely.geometry import Point, Polygon as Poly
    cat = train_catalogue[:4000]  # just a few data
    # synthetic_displacement = synthetic_displacement.reshape((3, 3 * synthetic_displacement.shape[2])).T
    synthetic_displacement = np.concatenate(
        (synthetic_displacement[0], synthetic_displacement[1], synthetic_displacement[2]), axis=1).T

    disp_facecolors = np.zeros((synthetic_displacement.shape[0],))
    if id_synth < 2:
        disp_facecolors[
        id_synth * (disp_facecolors.shape[0] // 3):(id_synth + 1) * (disp_facecolors.shape[0] // 3)] = 1.
    else:
        disp_facecolors[id_synth * (disp_facecolors.shape[0] // 3):] = 1.

    cascadia_box = [40 - 1.5, 51.8 - 0.2, -128.3 - 2.5, -121 + 2]  # min/max_latitude, min/max_longitude
    if not update:
        # fig, _ = plt.subplots(1, 1, figsize=(7.8, 7.8), dpi=100)
        fig = plt.figure(figsize=(7.8, 7.8), dpi=100)
        cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                               urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                               lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                               resolution='i', projection='lcc')  # , ax=ax9)

        cascadia_map.drawcoastlines(linewidth=0.5)
        cascadia_map.fillcontinents(color='bisque', lake_color='lightcyan')  # burlywood
        cascadia_map.drawmapboundary(fill_color='lightcyan')
        cascadia_map.drawmapscale(-122.5, 51.05, -122.5, 51.05, 300, barstyle='fancy', zorder=10)
        cascadia_map.drawparallels(np.arange(cascadia_box[0], cascadia_box[1], (cascadia_box[1] - cascadia_box[0]) / 4),
                                   labels=[0, 1, 0, 0], linewidth=0.1)
        cascadia_map.drawmeridians(np.arange(cascadia_box[2], cascadia_box[3], (cascadia_box[3] - cascadia_box[2]) / 4),
                                   labels=[0, 0, 0, 1], linewidth=0.1)
        cascadia_map.readshapefile('geo_data/tectonicplates-master/PB2002_boundaries', '', linewidth=0.3)

    if color_code_depth:
        xyz = read_from_slab2()[0]
        x, y, z = xyz[:, 0], xyz[:, 1], - xyz[:, 2]
        X_depth, Y_depth, Z_depth = grid(x, y, z, resX=300, resY=300)

        train_poly_x, train_poly_y = _train_contour_mod(cat, cascadia_map)
        train_poly = Poly(np.vstack((train_poly_x, train_poly_y)).T)
        mesh_points = np.vstack((X_depth.ravel(), Y_depth.ravel())).T
        flattened_Z = Z_depth.ravel()
        for i, point in enumerate(mesh_points):
            if not Point(point).within(train_poly):
                flattened_Z[i] = np.nan
        Z_depth = Z_depth.reshape(X_depth.shape)

        XX_depth, YY_depth = cascadia_map(X_depth, Y_depth)
        cmap_depth = plt.cm.get_cmap('turbo')
        pcm_depth = cascadia_map.pcolormesh(XX_depth, YY_depth, Z_depth, shading='gouraud', cmap=cmap_depth, alpha=1.,
                                            antialiased=True, rasterized=True)

        cbar_depth = plt.colorbar(pcm_depth, orientation='horizontal', fraction=0.018, pad=0.065, shrink=0.6,
                                  ticks=[10, 20, 30, 40, 50])  # fraction=0.046, pad=0.04)
        cbar_depth.ax.set_xlabel('Slab depth (km)', labelpad=5)

    if draw_dislocation:
        x_center = dislocation_parameters['epi_lon']
        y_center = dislocation_parameters['epi_lat']
        strike = np.deg2rad(dislocation_parameters['strike'])
        dip = np.deg2rad(dislocation_parameters['dip'])
        rake = np.deg2rad(dislocation_parameters['rake'])
        slip = dislocation_parameters['slip']
        L = dislocation_parameters['L']
        W = dislocation_parameters['W']
        d = dislocation_parameters['depth'] + np.sin(dip) * W / 2
        U1 = np.cos(rake) * slip
        U2 = np.sin(rake) * slip
        U3 = 0
        alpha = np.pi / 2 - strike
        x_fault = L / 2 * np.cos(alpha) * np.array([-1., 1., 1., -1.]) + np.sin(alpha) * np.cos(dip) * W / 2 * np.array(
            [-1., -1., 1., 1.])
        y_fault = L / 2 * np.sin(alpha) * np.array([-1., 1., 1., -1.]) + np.cos(alpha) * np.cos(dip) * W / 2 * np.array(
            [1., 1., -1., -1.])
        x_fault += x_center
        y_fault += y_center
        x_fault_map, y_fault_map = cascadia_map(x_fault, y_fault)
        # x_fault_map, y_fault_map = x_fault, y_fault
        plt.gca().add_patch(matplotlib.patches.Polygon(np.vstack((x_fault_map, y_fault_map)).T,
                                                       closed=True, fill=True, color=arrow_color, lw=1.5,
                                                       alpha=.4))
    if draw_gnss_network:
        cascadia_map.scatter(station_coordinates[:, 1], station_coordinates[:, 0], marker='^', s=15, color='C3',
                             latlon=True, edgecolors='black', linewidth=0.7)
    if draw_tremors:
        tremors = tremor_catalogue()
        db = DBSCAN(eps=1.5e-02, min_samples=15, n_jobs=-1).fit(tremors[:, :2])
        # joblib.dump(db, 'dbscan_tremors')
        # db = joblib.load('dbscan_tremors')
        denoised_tremors = np.where(db.labels_ != -1)[0]
        alpha_shape = alphashape.alphashape(tremors[denoised_tremors, :2], alpha=1.9)
        hull_pts = alpha_shape.exterior.coords.xy
        x_hull, y_hull = np.array(hull_pts[1]), np.array(hull_pts[0])
        valid_pts_hull = np.where(x_hull < -121.95)[0]  # remove knot
        x_hull, y_hull = x_hull[valid_pts_hull], y_hull[valid_pts_hull]
        dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 150)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        interp_x_map, interp_y_map = cascadia_map(interp_x, interp_y)
        cascadia_map.plot(interp_x_map, interp_y_map, linestyle='--', color='black', lw=2.)

    if draw_train_contour:
        '''alpha_shape = alphashape.alphashape(cat[:, :2], alpha=1.9)
        hull_pts = alpha_shape.exterior.coords.xy
        x_hull, y_hull = np.array(hull_pts[1]), np.array(hull_pts[0])
        # valid_pts_hull = np.where(x_hull < -121.95)[0]  # remove knot
        # x_hull, y_hull = x_hull[valid_pts_hull], y_hull[valid_pts_hull]
        dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
        interp_d_cat = np.linspace(dist_along[0], dist_along[-1], 350)
        interp_x_cat, interp_y_cat = interpolate.splev(interp_d_cat, spline)'''
        interp_x_cat, interp_y_cat = _train_contour_mod(cat, cascadia_map)
        interp_x_map_cat, interp_y_map_cat = cascadia_map(interp_x_cat, interp_y_cat)
        plt.gca().add_patch(matplotlib.patches.Polygon(np.vstack((interp_x_map_cat, interp_y_map_cat)).T,
                                                       closed=True, fill=False, color='C0', lw=2.,
                                                       alpha=.6))

    if draw_disp_field:
        max_length = np.max(np.sqrt(synthetic_displacement[:, 0] ** 2 + synthetic_displacement[:, 1] ** 2)) / 2
        # concatenate unit arrow
        unit_arrow_x, unit_arrow_y = -127., 39.5
        # x = np.concatenate((station_coordinates[:, 1], [unit_arrow_x]))
        # y = np.concatenate((station_coordinates[:, 0], [unit_arrow_y]))
        x = np.concatenate(
            (station_coordinates[:, 1], station_coordinates[:, 1], station_coordinates[:, 1], [unit_arrow_x]))
        y = np.concatenate(
            (station_coordinates[:, 0], station_coordinates[:, 0], station_coordinates[:, 0], [unit_arrow_y]))
        disp_x = np.concatenate((synthetic_displacement[:, 0], [-max_length]))
        disp_y = np.concatenate((synthetic_displacement[:, 1], [0.]))

        if not update:
            # cascadia_map.quiver(x, y, disp_x, disp_y, color=arrow_color, latlon=True, width=0.0035)
            colors = [arrow_color] * (len(x) - 1) + ['black']
            alpha = disp_facecolors.tolist() + [1]
            print(x.shape, y.shape, disp_x.shape, disp_y.shape, colors.__len__(), alpha.__len__())
            cascadia_map.quiver(x, y, disp_x, disp_y, color=colors, latlon=True,
                                width=0.0035, alpha=alpha)
        else:
            colors = [arrow_color] * (len(x) - 1)
            cascadia_map.quiver(x[:-1], y[:-1], disp_x[:-1], disp_y[:-1], color=colors, latlon=True, width=0.0035,
                                alpha=disp_facecolors)

        label_pos_x, label_pos_y = cascadia_map(unit_arrow_x - 1.45, unit_arrow_y + 0.25)
        if not update:
            plt.annotate(f'{np.around(max_length, decimals=1)} mm', xy=(unit_arrow_x, unit_arrow_y), xycoords='data',
                         xytext=(label_pos_x, label_pos_y), color='black')

    if draw_isodepth:
        def isodepth_label_fmt(x):
            return rf"{int(x)} km"

        levels = [20, 40]
        admissible_depth, _, _, region = read_from_slab2()
        x, y = admissible_depth[:, 0], admissible_depth[:, 1]
        depth = admissible_depth[:, 2]
        x_map, y_map = cascadia_map(x, y)
        isodepth = plt.tricontour(x_map, y_map, -depth, levels=levels, colors='black', linewidths=0.7)
        label_loc_y = [48.3, 41.7]  # [48.3, 48.4]
        label_loc_x = [-125.5, -123.0]  # [-125.5, -124.5]
        label_loc_x_map, label_loc_y_map = cascadia_map(label_loc_x, label_loc_y)
        plt.clabel(isodepth, isodepth.levels, inline=True, fontsize=10, fmt=isodepth_label_fmt,
                   manual=list(zip(label_loc_x_map, label_loc_y_map)))
    if final_update:
        ax = fig.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
        plt.savefig(f'{fig_directory}/map_synth.pdf', bbox_inches='tight')
        plt.close(fig)
    if not update:
        return cascadia_map, fig


def dislocation_parameters_from_catalogue(catalogue, disloc_idx):
    disloc_par = {'strike': catalogue[4][disloc_idx], 'dip': catalogue[5][disloc_idx], 'rake': catalogue[6][disloc_idx],
                  'depth': catalogue[2][disloc_idx], 'L': catalogue[7][disloc_idx] / 111.3194,
                  'W': catalogue[8][disloc_idx] / 111.3194, 'slip': catalogue[9][disloc_idx],
                  'epi_lat': catalogue[0][disloc_idx], 'epi_lon': catalogue[1][disloc_idx]}
    return disloc_par


if __name__ == '__main__':
    X, y, orig_cat, pred = _load_synth_data('predictions/pred_denoising_test_data', load_pred_only=False)
    noise = X[0] - y[0]
    cat = orig_cat[0]
    del X
    del y
    del pred

    print(noise.shape)

    station_codes, station_coordinates, _, _, _ = cascadia_filtered_stations(200)

    synth_parameters = {'new_stress_drop': True, 'magnitude_range': (6, 6.5), 'depth_range': (20, 40),
                        'rake_range': (80, 100)}  #new_stress_drop ONLY for figure, not in the paper
    transients, rand_dur, synth_disp, catalogue, single_trans = synthetic_sses_v2_mod(60, station_codes,
                                                                                      station_coordinates,
                                                                                      **synth_parameters)
    modeled_data = transients[0]
    synth_disp = np.array(synth_disp[0])
    single_trans = single_trans[0]
    print(catalogue[0])
    print(modeled_data.shape)

    vmin = None  # -0.3

    _draw_table_data(noise, [0, 60], station_coordinates, figtitle='noise', vmin=vmin, colorbar=True)
    _draw_table_data(modeled_data, [0, 60], station_coordinates, figtitle='model', ylabel=False, vmin=vmin,
                     colorbar=True)
    _draw_table_data(noise + modeled_data, [0, 60], station_coordinates, figtitle='noise+model', colorbar=True,
                     ylabel=False, vmin=vmin)

    for i in range(3):
        _draw_table_data(single_trans[i], [0, 60], station_coordinates, figtitle=f'model_{i + 1}', ylabel=False,
                         vmin=vmin, colorbar=True)

    # first dislocation
    disloc_idx = 0
    disloc_par = dislocation_parameters_from_catalogue(catalogue[0], disloc_idx)
    print(print(disloc_par))
    cascadia_map, fig = map_figure_synth(synth_disp, station_coordinates, disloc_par, orig_cat,
                                         draw_tremors=False, draw_gnss_network=True, draw_disp_field=True,
                                         draw_dislocation=True, draw_train_contour=True, draw_isodepth=True,
                                         color_code_depth=False, update=False, arrow_color='red', id_synth=disloc_idx)

    for idx in range(1, 3, 1):
        print(idx)
        disloc_par = dislocation_parameters_from_catalogue(catalogue[0], idx)
        print(disloc_par)
        map_figure_synth(synth_disp, station_coordinates, disloc_par, orig_cat,
                         draw_tremors=False, draw_gnss_network=False, draw_disp_field=True,
                         draw_dislocation=True, draw_train_contour=False, draw_isodepth=False,
                         color_code_depth=False, update=True, arrow_color=['green', 'blue'][idx - 1],
                         cascadia_map=cascadia_map, fig=fig, final_update=[False, True][idx - 1], id_synth=idx)

    # plt.show()

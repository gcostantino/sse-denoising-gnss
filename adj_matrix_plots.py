import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from tsl.nn.layers.graph_convs import AdaptiveGraphConv

from sse_denoiser.stagrnn import STAGRNNDenoiser


def adjacency_matrix_visualization(station_coordinates, every_nth_y=17, every_nth_x=37, window_length=60):
    params = {'n_stations': n_selected_stations,
              'window_length': window_length,
              'n_directions': 2,
              'batch_size': None,
              'station_coordinates': station_coordinates[:, :2],
              'y_test': None,
              'learn_static': False,
              'residual': False}

    denoiser = STAGRNNDenoiser(**params)
    denoiser.build()

    weight_path = 'weights/best_cascadia_02Jul2023-013725_train_denois_realgaps_v5_STAGRNN_no_CNN_old_sd.pt'
    denoiser.load_weights(weight_path)

    node_embedding = denoiser.get_model().agcrn.node_emb()
    adj_matrix = AdaptiveGraphConv.compute_adj(node_embedding).detach().numpy()
    '''plt.matshow(adj_matrix)
    plt.colorbar()
    plt.show()'''
    latsort = np.argsort(station_coordinates[:, 0])[::-1]
    print(station_coordinates[latsort])
    sorted_adj_matrix = adj_matrix[latsort][:, latsort]
    extent = [np.max(station_coordinates[:, 0]), np.min(station_coordinates[:, 0]), np.min(station_coordinates[:, 0]),
              np.max(station_coordinates[:, 0])]
    fig = plt.figure(figsize=(14, 7), dpi=300)
    # fig = plt.figure()
    plt.matshow(sorted_adj_matrix, extent=extent, cmap='gist_yarg')

    plt.gca().set_yticks(np.around(station_coordinates[latsort, 0], decimals=1))
    plt.gca().set_xticks(np.around(station_coordinates[latsort, 0], decimals=1))
    for n, label in enumerate(plt.gca().yaxis.get_ticklabels()):
        if n % every_nth_y != 0:
            label.set_visible(False)
            plt.gca().yaxis.get_major_ticks()[n].set_visible(False)

    for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if n % every_nth_x != 0:
            label.set_visible(False)
            plt.gca().xaxis.get_major_ticks()[n].set_visible(False)

    '''for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
            plt.gca().xaxis.get_major_ticks()[n].set_visible(False)'''
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().set(adjustable='box', aspect='equal')
    cbar.ax.set_ylabel('Edge strength', rotation=270, labelpad=25, size=13)
    plt.savefig('denoising_figures/adj_plots/adj_mat.pdf', bbox_inches='tight')
    plt.close(fig)
    # plt.show()

    fig = plt.figure(figsize=(8, 7), dpi=300)
    plt.hist(adj_matrix)
    plt.xlabel('Edge strength')
    plt.ylabel('Count')
    plt.savefig('denoising_figures/adj_plots/hist_adj.pdf', bbox_inches='tight')
    plt.close(fig)
    # plt.show()

    # group 1 --> between 0.008 to 0.0115
    # group 2 --> between 0.012 to 0.0155
    # group 3 --> between 0.016 to 0.0194
    # group 4 --> between 0.020 to 0.0234 # maybe group 3 and 4 can be grouped

    min_group_1 = 0.008
    max_group_1 = 0.0115

    min_group_2 = 0.012
    max_group_2 = 0.0155

    min_group_3 = 0.016
    max_group_3 = 0.0194

    min_group_4 = 0.020
    max_group_4 = 0.0234

    upper_triangular_adj_matrix = np.triu(adj_matrix)
    unique_edge_list_including_sloops = upper_triangular_adj_matrix[np.nonzero(upper_triangular_adj_matrix)].flatten()
    unique_edge_list_without_sloops = np.array(
        [upper_triangular_adj_matrix[i, j] for i in range(200) for j in range(200) if
         ((i != j) and (upper_triangular_adj_matrix[i, j] != 0))])

    # Analysis of the connections
    print('Number of total connections (including self-loops):', len(unique_edge_list_including_sloops))
    print('Number of weak connections (including self-loops):', np.sum(unique_edge_list_including_sloops < min_group_1))
    print('Number of "stronger" connections (including self-loops):',
          np.sum(unique_edge_list_including_sloops > min_group_1))

    print('Number of selected strong connections (0.008 < strength < 0.0234) (including self-loops):', np.sum(
        np.logical_and(unique_edge_list_including_sloops > min_group_1,
                       unique_edge_list_including_sloops < max_group_4)))

    print('---')

    print('Number of total connections (without self-loops):', len(unique_edge_list_without_sloops))
    print('Number of weak connections (without self-loops):', np.sum(unique_edge_list_without_sloops < min_group_1))
    print('Number of "stronger" connections (without self-loops):',
          np.sum(unique_edge_list_without_sloops > min_group_1))

    print('Number of selected strong connections (0.008 < strength < 0.0234) (without self-loops):', np.sum(
        [np.logical_and(unique_edge_list_without_sloops > min_group_1, unique_edge_list_without_sloops < max_group_4)]))

    # min_edge_filter = 0.006#0.005
    filtered_sorted_adj_matrix = sorted_adj_matrix.copy()
    filtered_sorted_adj_matrix[filtered_sorted_adj_matrix < min_group_1] = np.nan  # all together
    filtered_sorted_adj_matrix[filtered_sorted_adj_matrix > max_group_4] = np.nan

    fig = plt.figure(figsize=(14, 7), dpi=300)
    plt.matshow(filtered_sorted_adj_matrix, extent=extent, cmap='viridis')

    plt.gca().set_yticks(np.around(station_coordinates[latsort, 0], decimals=1))
    plt.gca().set_xticks(np.around(station_coordinates[latsort, 0], decimals=1))
    for n, label in enumerate(plt.gca().yaxis.get_ticklabels()):
        if n % every_nth_y != 0:
            label.set_visible(False)
            plt.gca().yaxis.get_major_ticks()[n].set_visible(False)

    for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if n % every_nth_x != 0:
            label.set_visible(False)
            plt.gca().xaxis.get_major_ticks()[n].set_visible(False)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().set(adjustable='box', aspect='equal')
    cbar.ax.set_ylabel('Edge strength', rotation=270, labelpad=25, size=13)
    plt.savefig('denoising_figures/adj_plots/filtered_adj_mat.pdf', bbox_inches='tight')
    plt.close(fig)
    # plt.show()

    filtered_unsorted_adj_matrix_nonan = filtered_sorted_adj_matrix.copy()
    filtered_unsorted_adj_matrix_nonan = filtered_unsorted_adj_matrix_nonan[np.argsort(latsort)][:, np.argsort(latsort)]
    filtered_unsorted_adj_matrix_nonan[np.isnan(filtered_unsorted_adj_matrix_nonan)] = 0.

    G = nx.from_numpy_matrix(filtered_unsorted_adj_matrix_nonan)

    fig = plt.figure(figsize=(14, 7), dpi=300)

    # cascadia_box = [40 - 1, 51.8 - 0.2, -128.3 - 1, -121 + 0.5 + 2]  # min/max_latitude, min/max_longitude
    cascadia_box = [40 - 0.2, 51.8 - 0.2 - 1, -128.3 - 1, -121 + 0.5 + 2]  # min/max_latitude, min/max_longitude
    cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                           urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                           lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                           resolution='i', projection='lcc')  # , ax=ax9)

    cascadia_map.drawcoastlines(linewidth=0.5)
    # cascadia_map.fillcontinents(color='bisque', lake_color='lightcyan')  # burlywood
    # cascadia_map.drawmapboundary(fill_color='lightcyan')
    # cascadia_map.drawmapscale(-122.5, 51.05, -122.5, 51.05, 300, barstyle='fancy', zorder=10)
    cascadia_map.drawparallels(np.arange(cascadia_box[0], cascadia_box[1], (cascadia_box[1] - cascadia_box[0]) / 4),
                               labels=[1, 0, 0, 0], linewidth=0.1)
    cascadia_map.drawmeridians(np.arange(cascadia_box[2], cascadia_box[3], (cascadia_box[3] - cascadia_box[2]) / 4),
                               labels=[0, 0, 0, 1], linewidth=0.1)
    cascadia_map.readshapefile('geo_data/tectonicplates-master/PB2002_boundaries', '', linewidth=0.3)

    x, y = cascadia_map(station_coordinates[:, 1], station_coordinates[:, 0])

    pos = {}
    for i in range(station_coordinates.shape[0]):
        pos[i] = (x[i], y[i])

    cmap = matplotlib.cm.get_cmap("turbo").copy()

    vmin = np.nanmin(filtered_sorted_adj_matrix)
    vmax = np.nanmax(filtered_sorted_adj_matrix)

    edge_weights = [w for u, v, w in G.edges(data='weight')]

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    width = 0.5
    '''nx.draw_networkx(G, pos, with_labels=False, node_size=5, node_color='black', edge_color=cmap(norm(edge_weights)),
                    width=width, cmap=cmap)'''

    nx.draw_networkx(G, pos, with_labels=False, node_size=5, node_color='black', edge_color=cmap(norm(edge_weights)),
                     nodelist=[], width=width, cmap=cmap)

    cascadia_map.scatter(station_coordinates[:, 1], station_coordinates[:, 0], marker='^', s=10, color='C3',
                         latlon=True, edgecolors='black', linewidth=0.7)

    # plt.title(f'Learnt adjacency matrix')
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), fraction=0.046,
                        pad=0.04)  # plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Edge strength', rotation=270, labelpad=25, size=13)
    plt.savefig('denoising_figures/adj_plots/map_strong_conections.pdf', bbox_inches='tight')
    plt.close(fig)
    # plt.show()

    # station importance plot

    self_loop_strength = np.diag(adj_matrix)
    fig = plt.figure(figsize=(14, 7), dpi=300)
    cascadia_map = Basemap(llcrnrlon=cascadia_box[2], llcrnrlat=cascadia_box[0],
                           urcrnrlon=cascadia_box[3], urcrnrlat=cascadia_box[1],
                           lat_0=cascadia_box[0], lon_0=cascadia_box[2],
                           resolution='i', projection='tmerc')
    # cascadia_map.drawmapboundary(fill_color='aqua')
    # cascadia_map.fillcontinents(color='grey', lake_color='aqua')
    cascadia_map.drawcoastlines()

    cascadia_map.drawparallels(np.arange(cascadia_box[0], cascadia_box[1], (cascadia_box[1] - cascadia_box[0]) / 4),
                               labels=[1, 0, 0, 0])
    cascadia_map.drawmeridians(np.arange(cascadia_box[2], cascadia_box[3], (cascadia_box[3] - cascadia_box[2]) / 4),
                               labels=[0, 0, 0, 1])
    cascadia_map.readshapefile('geo_data/tectonicplates-master/PB2002_boundaries', '', linewidth=0.3)
    # cascadia_map.scatter(station_coordinates[~np.isnan(np.diag(adj_matrix)), 1], station_coordinates[~np.isnan(np.diag(adj_matrix)), 0], c=adj_matrix[~np.isnan(np.diag(adj_matrix)), ~np.isnan(np.diag(adj_matrix))], latlon=True)
    sc = cascadia_map.scatter(station_coordinates[:, 1], station_coordinates[:, 0], c=self_loop_strength, cmap='turbo',
                              latlon=True)
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)  # plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Station importance', rotation=270, labelpad=25, size=13)
    plt.savefig('denoising_figures/adj_plots/station_importance.pdf', bbox_inches='tight')
    plt.close(fig)
    # plt.show()


if __name__ == '__main__':
    n_selected_stations = 200

    # station_codes, station_coordinates, _, _, _ = cascadia_filtered_stations(n_selected_stations)
    with np.load('denoised_ts.npz') as f:
        station_coordinates = f['coordinates']

    adjacency_matrix_visualization(station_coordinates)

"""Generate a time space diagram for some networks.

This method accepts as input a csv file containing the sumo-formatted emission
file, and then uses this data to generate a time-space diagram, with the x-axis
being the time (in seconds), the y-axis being the position of a vehicle, and
color representing the speed of te vehicles.

If the number of simulation steps is too dense, you can plot every nth step in
the plot by setting the input `--steps=n`.

Note: This script assumes that the provided network has only one lane on the
each edge, or one lane on the main highway in the case of MergeNetwork.

Usage
-----
::
    python time_space_diagram.py </path/to/emission>.csv </path/to/params>.json
"""
from flow.utils.rllib import get_flow_params
from flow.networks import RingNetwork, FigureEightNetwork, MergeNetwork,\
    I210SubNetwork, I24SubNetwork, HighwayNetwork

import argparse
from collections import defaultdict
try:
    from matplotlib import pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import numpy as np
import pandas as pd


# networks that can be plotted by this method
ACCEPTABLE_NETWORKS = [
    RingNetwork,
    FigureEightNetwork,
    MergeNetwork,
    I210SubNetwork,
    I24SubNetwork,
    HighwayNetwork
]

# networks that use edgestarts
USE_EDGESTARTS = set([
    RingNetwork,
    FigureEightNetwork,
    MergeNetwork,
    I24SubNetwork
])

GHOST_DICT = defaultdict(dict)
GHOST_DICT[I210SubNetwork] = {'ghost_edges': {'ghost0', '119257908#3'}}
GHOST_DICT[I24SubNetwork] = {'ghost_edges': {'Westbound_2', 'Westbound_7'}}
GHOST_DICT[HighwayNetwork] = {'ghost_bounds': (500, 2300)}


def import_data_from_trajectory(fp, params=dict()):
    r"""Import and preprocess data from the Flow trajectory (.csv) file.

    Parameters
    ----------
    fp : str
        file path (for the .csv formatted file)
    params : dict
        flow-specific parameters, including:
        * "network" (str): name of the network that was used when generating
          the emission file. Must be one of the network names mentioned in
          ACCEPTABLE_NETWORKS,
        * "net_params" (flow.core.params.NetParams): network-specific
          parameters. This is used to collect the lengths of various network
          links.

    Returns
    -------
    pd.DataFrame, float, float
    """
    network = params['network']

    # Read trajectory csv into pandas dataframe
    df = pd.read_csv(fp)

    # Convert column names for backwards compatibility using emissions csv
    column_conversions = {
        'time': 'time_step',
        'lane_number': 'lane_id',
    }
    df = df.rename(columns=column_conversions)
    if network in USE_EDGESTARTS:
        df['distance'] = _get_abs_pos(df, params)
    start = params['env'].warmup_steps * params['env'].sims_per_step * params['sim'].sim_step
    # produce upper and lower bounds for the non-greyed-out domain
    ghost_edges = GHOST_DICT[network].get('ghost_edges')
    ghost_bounds = GHOST_DICT[network].get('ghost_bounds')
    if ghost_edges:
        main_ids = df[df['edge_id'].isin(ghost_edges)]['id'].unique()
        domain_lb = df[(df['id'].isin(main_ids)) & (~df['edge_id'].isin(ghost_edges))]['distance'].min()
        domain_ub = df[(df['id'].isin(main_ids)) & (~df['edge_id'].isin(ghost_edges))]['distance'].max()
    elif ghost_bounds:
        domain_lb = ghost_bounds[0]
        domain_ub = ghost_bounds[1]
    else:
        domain_lb = df['distance'].min()
        domain_ub = df['distance'].max()

    df.loc[:, 'time_step'] = df['time_step'].apply(lambda x: x - start)
    df.loc[:, 'distance'] = df['distance'].apply(lambda x: x - domain_lb)
    domain_ub -= domain_lb

    # Compute line segment ends by shifting dataframe by 1 row
    df[['next_pos', 'next_time']] = df.groupby('id')[['distance', 'time_step']].shift(-1)

    # Remove nans from data
    df = df[df['next_time'].notna()]

    return df, domain_lb, domain_ub, start


def get_time_space_data(data, network):
    r"""Compute the unique inflows and subsequent outflow statistics.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data
    network : child class of Network()
        network that was used when generating the emission file.
        Must be one of the network names mentioned in
        ACCEPTABLE_NETWORKS

    Returns
    -------
    ndarray (or dict < str, np.ndarray >)
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
        in the case of I210, the nested arrays are wrapped into a dict,
        keyed on the lane number, so that each lane can be plotted
        separately.

    Raises
    ------
    AssertionError
        if the specified network is not supported by this method
    """
    # check that the network is appropriate
    assert network in ACCEPTABLE_NETWORKS, \
        'Network must be one of: ' + ', '.join([network_.__name__ for network_ in ACCEPTABLE_NETWORKS])

    # switcher used to compute the positions based on the type of network
    switcher = {
        RingNetwork: _ring_road,
        MergeNetwork: _merge,
        FigureEightNetwork: _figure_eight,
        I210SubNetwork: _i210_subnetwork,
        I24SubNetwork: _i24_subnetwork,
        HighwayNetwork: _highway,
    }

    # Get the function from switcher dictionary
    func = switcher[network]

    # Execute the function
    segs, data = func(data)

    return segs, data


def _merge(data):
    r"""Generate time and position data for the merge.

    This only include vehicles on the main highway, and not on the adjacent
    on-ramp.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data

    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        modified trajectory dataframe
    """
    # Omit ghost edges
    keep_edges = {'inflow_merge', 'bottom', ':bottom_0'}
    data = data[data['edge_id'].isin(keep_edges)]

    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _highway(data):
    r"""Generate time and position data for the highway.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data

    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        modified trajectory dataframe
    """
    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _ring_road(data):
    r"""Generate time and position data for the ring road.

    Vehicles that reach the top of the plot simply return to the bottom and
    continue.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data

    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        unmodified trajectory dataframe
    """
    data = data[data['next_pos'] > data['distance']]
    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _i210_subnetwork(data):
    r"""Generate time and position data for the i210 subnetwork.

    We generate plots for all lanes, so the segments are wrapped in
    a dictionary.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data

    Returns
    -------
    dict < str, np.ndarray >
        dictionary of 3d array (n_segments x 2 x 2) containing segments
        to be plotted. the dictionary is keyed on lane numbers, with the
        values being the 3d array representing the segments. every inner
        2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        modified trajectory dataframe
    """
    # Reset lane numbers that are offset by ramp lanes
    offset_edges = set(data[data['lane_id'] == 5]['edge_id'].unique())
    data.loc[~data['edge_id'].isin(offset_edges), 'lane_id'] = \
        data[~data['edge_id'].isin(offset_edges)]['lane_id'] + 1

    segs = dict()
    for lane, df in data.groupby('lane_id'):
        segs[lane] = df[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(df), 2, 2))

    return segs, data


def _i24_subnetwork(data):
    r"""Generate time and position data for the i24 subnetwork.

    We generate plots for all lanes, so the segments are wrapped in
    a dictionary.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data

    Returns
    -------
    dict < str, np.ndarray >
        dictionary of 3d array (n_segments x 2 x 2) containing segments
        to be plotted. the dictionary is keyed on lane numbers, with the
        values being the 3d array representing the segments. every inner
        2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        modified trajectory dataframe
    """
    # Reset lane numbers that are offset by ramp lanes
    ramp_edges = set([
        'Westbound_On_1',
        ':202185444_0',
        'Westbound_4',
        ':202192541_0',
        'Westbound_Off_2'])
    data.loc[~data['edge_id'].isin(ramp_edges), 'lane_id'] = \
        data[~data['edge_id'].isin(ramp_edges)]['lane_id'] + 1

    segs = dict()
    for lane, df in data.groupby('lane_id'):
        segs[lane] = df[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(df), 2, 2))

    return segs, data


def _figure_eight(data):
    r"""Generate time and position data for the figure eight.

    The vehicles traveling towards the intersection from one side will be
    plotted from the top downward, while the vehicles from the other side will
    be plotted from the bottom upward.

    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data

    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        unmodified trajectory dataframe
    """
    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _get_abs_pos(df, params):
    """Compute the absolute positions from edges and relative positions.

    This is the variable we will ultimately use to plot individual vehicles.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of trajectory data
    params : dict
        flow-specific parameters

    Returns
    -------
    pd.Series
        the absolute positive for every sample
    """
    if params['network'] == MergeNetwork:
        inflow_edge_len = 100
        premerge = params['net'].additional_params['pre_merge_length']
        postmerge = params['net'].additional_params['post_merge_length']

        # generate edge starts
        edgestarts = {
            'inflow_highway': 0,
            'left': inflow_edge_len + 0.1,
            'center': inflow_edge_len + premerge + 22.6,
            'inflow_merge': inflow_edge_len + premerge + postmerge + 22.6,
            'bottom': 2 * inflow_edge_len + premerge + postmerge + 22.7,
            ':left_0': inflow_edge_len,
            ':center_0': inflow_edge_len + premerge + 0.1,
            ':center_1': inflow_edge_len + premerge + 0.1,
            ':bottom_0': 2 * inflow_edge_len + premerge + postmerge + 22.6
        }
    elif params['network'] == RingNetwork:
        ring_length = params['net'].additional_params["length"]
        junction_length = 0.1  # length of inter-edge junctions

        edgestarts = {
            "bottom": 0,
            ":right_0": 0.25 * ring_length,
            "right": 0.25 * ring_length + junction_length,
            ":top_0": 0.5 * ring_length + junction_length,
            "top": 0.5 * ring_length + 2 * junction_length,
            ":left_0": 0.75 * ring_length + 2 * junction_length,
            "left": 0.75 * ring_length + 3 * junction_length,
            ":bottom_0": ring_length + 3 * junction_length
        }
    elif params['network'] == FigureEightNetwork:
        net_params = params['net']
        ring_radius = net_params.additional_params['radius_ring']
        ring_edgelen = ring_radius * np.pi / 2.
        intersection = 2 * ring_radius
        junction = 2.9 + 3.3 * net_params.additional_params['lanes']
        inner = 0.28

        # generate edge starts
        edgestarts = {
            'bottom': inner,
            'top': intersection / 2 + junction + inner,
            'upper_ring': intersection + junction + 2 * inner,
            'right': intersection + 3 * ring_edgelen + junction + 3 * inner,
            'left': 1.5 * intersection + 3 * ring_edgelen + 2 * junction + 3 * inner,
            'lower_ring': 2 * intersection + 3 * ring_edgelen + 2 * junction + 4 * inner,
            ':bottom_0': 0,
            ':center_1': intersection / 2 + inner,
            ':top_0': intersection + junction + inner,
            ':right_0': intersection + 3 * ring_edgelen + junction + 2 * inner,
            ':center_0': 1.5 * intersection + 3 * ring_edgelen + junction + 3 * inner,
            ':left_0': 2 * intersection + 3 * ring_edgelen + 2 * junction + 3 * inner,
            # for aimsun
            'bottom_to_top': intersection / 2 + inner,
            'right_to_left': junction + 3 * inner,
        }
    elif params['network'] == HighwayNetwork:
        return df['x']
    elif params['network'] == I210SubNetwork:
        edgestarts = {
            '119257914': -5.0999999999995795,
            '119257908#0': 56.49000000018306,
            ':300944379_0': 56.18000000000016,
            ':300944436_0': 753.4599999999871,
            '119257908#1-AddedOnRampEdge': 756.3299999991157,
            ':119257908#1-AddedOnRampNode_0': 853.530000000022,
            '119257908#1': 856.7699999997207,
            ':119257908#1-AddedOffRampNode_0': 1096.4499999999707,
            '119257908#1-AddedOffRampEdge': 1099.6899999995558,
            ':1686591010_1': 1198.1899999999541,
            '119257908#2': 1203.6499999994803,
            ':1842086610_1': 1780.2599999999056,
            '119257908#3': 1784.7899999996537,
        }
    elif params['network'] == I24SubNetwork:
        edgestarts = {
            'Westbound_2': -4.600000000000193,
            ':202177572_1': 392.4199999999998,
            'Westbound_3': 415.61999999999955,
            'Westbound_On_1': 635.1099999999994,
            ':202185444_1': 961.7699999999996,
            ':202185444_0': 963.0999999999995,
            'Westbound_4': 987.3099999999993,
            ':202192541_0': 1527.2599999999993,
            ':202192541_1': 1527.2599999999993,
            'Westbound_Off_2': 1540.5999999999992,
            'Westbound_5': 1540.7299999999993,
            ':306535021_0': 1688.1499999999996,
            'Westbound_6': 1688.4899999999993,
            ':306535022_0': 1738.4099999999999,
            'Westbound_7': 1738.8399999999992,
        }
    else:
        edgestarts = defaultdict(float)

    df = df[df['edge_id'].notna()]
    ret = df.apply(lambda x: x['relative_position'] + edgestarts[x['edge_id']], axis=1)

    if params['network'] == FigureEightNetwork:
        # reorganize data for space-time plot
        figure_eight_len = 6 * ring_edgelen + 2 * intersection + 2 * junction + 10 * inner
        intersection_loc = [edgestarts[':center_1'] + intersection / 2,
                            edgestarts[':center_0'] + intersection / 2]
        ret.loc[ret < intersection_loc[0]] += figure_eight_len
        ret.loc[(ret > intersection_loc[0]) & (ret < intersection_loc[1])] += -intersection_loc[1]
        ret.loc[ret > intersection_loc[1]] = \
            - ret.loc[ret > intersection_loc[1]] + figure_eight_len + intersection_loc[0]
    return ret


def plot_tsd(df, network, cmap, min_speed=0, max_speed=10, start=0, domain_bounds=None):
    """Plot the time-space diagram.

    Take the pre-processed segments and other meta-data, then plot all the line
    segments.

    Parameters
    ----------
    df : pd.DataFrame
        data used for axes bounds and speed coloring
    network : child class of Network()
        network that was used when generating the emission file.
        Must be one of the network names mentioned in
        ACCEPTABLE_NETWORKS
    cmap : colors.LinearSegmentedColormap
        colormap for plotting speed
    min_speed : int or float
        minimum speed in colorbar
    max_speed : int or float
        maximum speed in colorbar
    start : int or float
        starting time_step not greyed out
    domain_bounds : tuple
        lower and upper bounds of domain, excluding ghost edges, default None
    """
    norm = plt.Normalize(min_speed, max_speed)

    xmin, xmax = df['time_step'].min(), df['time_step'].max()
    xbuffer = (xmax - xmin) * 0.025  # 2.5% of range
    ymin, ymax = df['distance'].min(), df['distance'].max()
    ybuffer = (ymax - ymin) * 0.025  # 2.5% of range

    # Convert df data into segments for plotting
    segs, df = get_time_space_data(df, network)

    nlanes = df['lane_id'].nunique()
    plt.figure(figsize=(16, 9*nlanes))
    if nlanes == 1:
        segs = [segs]

    lane_count = 0
    for lane, lane_df in df.groupby('lane_id'):
        lane_count += 1
        ax = plt.subplot(nlanes, 1, lane_count)

        lc = LineCollection(segs[lane], cmap=cmap, norm=norm)
        lc.set_array(lane_df['speed'].values)
        lc.set_linewidth(1)
        ax.add_collection(lc)
        ax.autoscale()

        rects = []
        # rectangle for warmup period, but not ghost edges
        rects.append(Rectangle((xmin, 0), start, domain_bounds[1]))
        # rectangle for lower ghost edge (including warmup period)
        rects.append(Rectangle((xmin, ymin), xmax - xmin, domain_bounds[0]))
        # rectangle for upper ghost edge (including warmup period)
        rects.append(Rectangle((xmin, domain_bounds[1]), xmax - xmin, ymax - domain_bounds[1]))

        pc = PatchCollection(rects, facecolor='grey', alpha=0.5, edgecolor=None)
        pc.set_zorder(20)
        ax.add_collection(pc)

        if nlanes > 1:
            if lane == 0:
                ax.set_title('Time-Space Diagram: Ramp/Merge Lane', fontsize=25)
            else:
                ax.set_title('Time-Space Diagram: Lane {}'.format(lane), fontsize=25)
        else:
            ax.set_title('Time-Space Diagram', fontsize=25)

        ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
        ax.set_ylim(ymin - ybuffer, ymax + ybuffer)

        ax.set_ylabel('Position (m)', fontsize=20)
        if lane_count == nlanes:
            ax.set_xlabel('Time (s)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        cbar = plt.colorbar(lc, ax=ax, norm=norm)
        cbar.set_label('Velocity (m/s)', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()


def tsd_main(trajectory_path, flow_params, min_speed=0, max_speed=10):
    """Prepare and plot the time-space diagram.

    Parameters
    ----------
    trajectory_path : str
        file path (for the .csv formatted file)
    flow_params : dict
        flow-specific parameters, including:
        * "network" (str): name of the network that was used when generating
          the emission file. Must be one of the network names mentioned in
          ACCEPTABLE_NETWORKS,
        * "net_params" (flow.core.params.NetParams): network-specific
          parameters. This is used to collect the lengths of various network
          links.
    min_speed : int or float
        minimum speed in colorbar
    max_speed : int or float
        maximum speed in colorbar
    """
    network = flow_params['network']

    # some plotting parameters
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    # Read trajectory csv into pandas dataframe
    traj_df, domain_lb, domain_ub, start = import_data_from_trajectory(trajectory_path, flow_params)

    plot_tsd(df=traj_df,
             network=network,
             cmap=my_cmap,
             min_speed=min_speed,
             max_speed=max_speed,
             start=start,
             domain_bounds=(domain_lb, domain_ub))

    ###########################################################################
    #                       Note: For MergeNetwork only                       #
    if network == MergeNetwork:                                               #
        plt.plot([traj_df['time_step'].min(), traj_df['time_step'].max()],
                 [0, 0], linewidth=3, color="white")                          #
        plt.plot([traj_df['time_step'].min(), traj_df['time_step'].max()],
                 [-0.1, -0.1], linewidth=3, color="white")                    #
    ###########################################################################

    outfile = trajectory_path.replace('csv', 'png')
    plt.savefig(outfile)


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Generates time space diagrams for flow networks.',
        epilog='python time_space_diagram.py </path/to/emission>.csv '
               '</path/to/flow_params>.json')

    # required arguments
    parser.add_argument('trajectory_path', type=str,
                        help='path to the Flow trajectory csv file.')
    parser.add_argument('flow_params', type=str,
                        help='path to the flow_params json file.')

    # optional arguments
    parser.add_argument('--steps', type=int, default=1,
                        help='rate at which steps are plotted.')
    parser.add_argument('--title', type=str, default='Time Space Diagram',
                        help='Title for the time-space diagrams.')
    parser.add_argument('--max_speed', type=int, default=8,
                        help='The maximum speed in the color range.')
    parser.add_argument('--min_speed', type=int, default=0,
                        help='The minimum speed in the color range.')
    args = parser.parse_args()

    # flow_params is imported as a dictionary
    if '.json' in args.flow_params:
        flow_params = get_flow_params(args.flow_params)
    else:
        module = __import__("examples.exp_configs.non_rl", fromlist=[args.flow_params])
        flow_params = getattr(module, args.flow_params).flow_params

    tsd_main(
        args.trajectory_path,
        flow_params,
        min_speed=args.min_speed,
        max_speed=args.max_speed
    )

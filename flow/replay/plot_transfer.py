"""Plot util for transfer tests."""
try:
    from matplotlib import pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt

import argparse
import numpy as np
import csv
import os

EXAMPLE_USAGE = """
example usage:
    python plot_transfer.py /path/to/data --output_dir /some/random/dir --full --exp_title moqef_0
        --save_fig --save_separate
    python plot_transfer.py /path/to/data /second/path/to/data --output_dir /path/to/data
        --plot_idm --plot_outflow --show_plot --exp_title moqef_0

Here the arguments are:
1 - the path to the simulation results
2+ - keyword arguments, see parser initialization
"""


# Paths, hardcoded
BASE = ''
IDM = 'idm_sweep'
INFLOW = 'inflow_sweep'
LANEFREQ = 'lanefreq_sweep'
OUTFLOW = 'outflow_sweep'


# TODO, base values of sweep
IDM_VAL = 1.3
INFLOW_VAL = 2050
LANEFREQ_VAL = 1
OUTFLOW_VAL = 5


def plot(args):
    """Plot the data."""
    data = {}
    paths = []
    if args.full:
        paths = [IDM, INFLOW, LANEFREQ, OUTFLOW]
    else:
        if args.plot_idm:
            paths.append(IDM)
        if args.plot_inflow:
            paths.append(INFLOW)
        if args.plot_lanefreq:
            paths.append(LANEFREQ)
        if args.plot_outflow:
            paths.append(OUTFLOW)

    # Collect the data from the CSV file
    for directory in args.directory:
        for path in paths:
            d = []
            with open(os.path.join(directory, path, 'data.csv')) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print('Column names are {}'.format(", ".join(row)))
                        line_count += 1
                    else:
                        d.append([float(x) for x in row])
                        line_count += 1
                print('Processed {} lines.'.format(line_count))
            data[directory + path] = np.array(d)

    def add_subplot(fig, loc, path, index, x_vals, title, x_label, y_label, x_tick=None):
        ax = fig.add_subplot(loc)
        for i, d in enumerate(args.directory):
            if args.labels:
                ax.plot(x_vals, data[d + path][:, index], label=args.labels[i])
            else:
                ax.plot(x_vals, data[d + path][:, index])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        if x_tick:
            plt.axvspan(x_tick, x_tick, color='red', alpha=0.5)
        ax.title.set_text(title)
        return ax

    def plot_individual(path, index, x_vals, title, x_label, y_label, x_tick):
        for i, d in enumerate(args.directory):
            if args.labels:
                plt.plot(x_vals, data[d + path][:, index], label=args.labels[i])
            else:
                plt.plot(x_vals, data[d + path][:, index])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        if x_tick:
            plt.axvspan(x_tick, x_tick, color='red', alpha=0.5)
        plt.title(title)
        output_path = os.path.join(args.output_dir, path)
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, title + '_transfer.png'))
        plt.close()

    for path in paths:
        if path == IDM:
            x_vals = [0.2, 0.6, 1.0, 1.4, 1.8]
            x_tick = IDM_VAL
            x_label = "IDM max accel"
        elif path == INFLOW:
            x_vals = [1800, 2000, 2200, 2400]
            x_tick = INFLOW_VAL
            x_label = "Inflow rate"
        elif path == LANEFREQ:
            x_vals = [1, 10, 20]
            x_tick = LANEFREQ_VAL
            x_label = "lcSpeedGain"
        elif path == OUTFLOW:
            x_vals = [2, 4, 6, 8]
            x_tick = OUTFLOW_VAL
            x_label = "Outflow speed"

        fig = plt.figure()
        fig.subplots_adjust(wspace=1.0)
        fig.subplots_adjust(hspace=1.5)

        add_subplot(fig, 331, path, 0, x_vals, 'velocity', x_label, 'velocity', x_tick)
        add_subplot(fig, 332, path, 1, x_vals, '# outflows', x_label, '# outflows', x_tick)
        add_subplot(fig, 333, path, 2, x_vals, 'energy', x_label, 'energy', x_tick)
        add_subplot(fig, 334, path, 3, x_vals, 'trip time', x_label, 'trip time', x_tick)
        add_subplot(fig, 335, path, 5, x_vals, 'inst mpg', x_label, 'inst mpg', x_tick)
        add_subplot(fig, 336, path, 6, x_vals, 'accel', x_label, 'accel', x_tick)
        add_subplot(fig, 337, path, 7, x_vals, 'headway', x_label, 'headway', x_tick)
        add_subplot(fig, 338, path, 12, x_vals, 'lane count', x_label, 'lane count', x_tick)

        fig.suptitle("{} for {}".format(path, args.exp_title))
        plt.legend(loc='upper center', bbox_to_anchor=(2.45, 0.85), shadow=True, ncol=1)
        if args.save_fig:
            output_path = os.path.join(args.output_dir, path)
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, 'transfer.png'))
        if args.show_plot:
            plt.show()

        if args.save_separate:
            plot_individual(path, 0, x_vals, 'velocity', x_label, 'velocity', x_tick)
            plot_individual(path, 1, x_vals, '# outflows', x_label, '# outflows', x_tick)
            plot_individual(path, 2, x_vals, 'energy', x_label, 'energy', x_tick)
            plot_individual(path, 3, x_vals, 'trip time', x_label, 'trip time', x_tick)
            plot_individual(path, 5, x_vals, 'inst mpg', x_label, 'inst mpg', x_tick)
            plot_individual(path, 6, x_vals, 'accel', x_label, 'accel', x_tick)
            plot_individual(path, 7, x_vals, 'headway', x_label, 'headway', x_tick)
            plot_individual(path, 12, x_vals, 'lane count', x_label, 'lane count', x_tick)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Plotting device for i210 transfer tests.',
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        'directory',
        nargs='+',
        type=str,
        help='Directories containing results'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        type=str,
        default=None,
        help='Labels in corresponding orders to directories listed'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to output results'
    )
    parser.add_argument(
        '--exp_title',
        required=True,
        help='Name of policy or plot'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Whether or not to run the full suite of plots.'
    )
    parser.add_argument(
        '--save_fig',
        action='store_true',
        help='Whether or not to save a png of the plots.'
    )
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help='Whether or not to run plt.show()'
    )
    parser.add_argument(
        '--plot_idm', '-i',
        action='store_true',
        help='Bool to plot the results of the IDM sweep.'
             ' Overwritten by --full'
    )
    parser.add_argument(
        '--plot_outflow', '-o',
        action='store_true',
        help='Bool to plot the results of the outflow sweep.'
             ' Overwritten by --full'
    )
    parser.add_argument(
        '--plot_lanefreq', '-l',
        action='store_true',
        help='Bool to plot the results of the lane frequeny sweep.'
             ' Overwritten by --full'
    )
    parser.add_argument(
        '--plot_inflow', '-if',
        action='store_true',
        help='Bool to plot the results of the inflow sweep.'
             ' Overwritten by --full'
    )
    parser.add_argument(
        '--save_separate',
        action='store_true',
        help='Save all the plots separately'
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    plot(args)

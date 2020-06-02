"""Plot training history"""

import argparse
import os

import pandas as pd

from lstm_ee.plot.history import plot_train_val_history

def load_train_log(savedir):
    """Load training history"""
    return pd.read_csv("%s/log.csv" % (savedir))

def parse_cmdargs():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("Train history plotter")

    parser.add_argument(
        'savedir',
        metavar = 'MODEL_DIR',
        type    = str,
        help    = 'Model save directory'
    )

    parser.add_argument(
        '-e', '--ext',
        dest    = 'ext',
        default = 'png',
        help    = 'Plot file extension',
        nargs   = '+',
        type    = str,
    )

    parser.add_argument(
        '-x',
        choices = [ 'epoch', 'train_time' ],
        default = 'epoch',
        dest    = 'x',
        help    = 'X axis',
        type    = str,
    )

    parser.add_argument(
        '-y',
        choices = [
            'loss',
            'target_total_loss',
            'target_primary_loss',
            'target_total_ms_relative_error',
            'target_total_mean_relative_error',
            'target_primary_ms_relative_error',
            'target_primary_mean_relative_error',
        ],
        default = 'loss',
        dest    = 'y',
        help    = 'Y axis',
        type    = str,
    )

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
    cmdargs = parse_cmdargs()

    log = load_train_log(cmdargs.savedir)

    plotdir = os.path.join(cmdargs.savedir, 'plots')
    os.makedirs(plotdir, exist_ok = True)

    print("Starting history plots script:")

    # How many epochs skip before plotting
    skip_list  = [ 0, 1, 2 ,3 ]
    skip_list += [ int(len(log) * x) for x in [0.25, 0.50, 0.75] ]

    for skip in skip_list:
        print(" * Making history plot of %s vs %s, skip %d..."
              % (cmdargs.y, cmdargs.x, skip)
        )
        plot_train_val_history(
            cmdargs.x, cmdargs.y, log, skip, plotdir, cmdargs.ext
        )

if __name__ == '__main__':
    main()


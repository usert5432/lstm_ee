"""Visualize `keras` model."""

import argparse
import os

from keras.utils      import plot_model
from lstm_ee.utils.io import load_model

def parse_cmdargs():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("Print model")

    parser.add_argument(
        'outdir',
        help    = 'Directory with saved models',
        metavar = 'OUTDIR',
        type    = str,
    )

    parser.add_argument(
        '-e', '--ext',
        default = 'png',
        dest    = 'ext',
        help    = 'Plot file extension',
        type    = str,
    )

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
    cmdargs  = parse_cmdargs()
    _, model = load_model(cmdargs.outdir, compile = False)

    plotdir = os.path.join(cmdargs.outdir, 'plots')
    os.makedirs(plotdir, exist_ok = True)

    plot_model(
        model,
        show_shapes      = True,
        show_layer_names = False,
        rankdir          = 'LR',
        to_file          = '%s/model.%s' % (plotdir, cmdargs.ext)
    )

if __name__ == '__main__':
    main()


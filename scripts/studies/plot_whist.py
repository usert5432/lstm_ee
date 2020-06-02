"""Plot inverse of TrueE histogram that is used to calculate flat weights"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot  import plot_nphist1d_base, save_fig

from lstm_ee.args        import Args
from lstm_ee.data        import load_data
from lstm_ee.utils.eval  import EvalConfig, make_eval_outdir, make_plotdir
from lstm_ee.utils.log   import setup_logging

from lstm_ee.data.data_generator.funcs.weights import calc_flat_whist

def parse_cmdargs():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("Plot flat weights hist")
    parser.add_argument(
        'outdir',
        help    = 'Directory with saved models',
        metavar = 'OUTDIR',
        type    = str,
    )

    parser.add_argument(
        '-e', '--ext',
        help    = 'Plot file extension',
        default = [ 'png' ],
        dest    = 'ext',
        nargs   = '+',
        type    = str,
    )

    parser.add_argument(
        '-d', '--data',
        help    = 'Evaluate on a different dataset',
        default = None,
        dest    = 'data',
        type    = str,
    )

    cmdargs = parser.parse_args()

    cmdargs.weights   = 'same'
    cmdargs.test_size = 'same'
    cmdargs.preset    = 'none'
    cmdargs.noise     = 'none'

    return cmdargs

def get_weights_hist_func(args):
    """Get function that calculates inverse of the TrueE histogram"""

    if (args.weights is None) or (args.weights['name'] != 'flat'):
        raise ValueError("Invalid weight: %s" % (args.weights))

    # pylint: disable=unnecessary-lambda
    return lambda x : calc_flat_whist(x, **args.weights['kwargs'])

def prologue():
    """Load data, model and setup output directory"""
    setup_logging()
    cmdargs = parse_cmdargs()

    args = Args.load(savedir = cmdargs.outdir)
    eval_config = EvalConfig.from_cmdargs(cmdargs)
    eval_config.modify_eval_args(args)

    dgen_train, dgen_test = load_data(args)

    outdir  = make_eval_outdir(cmdargs.outdir, eval_config)
    outdir  = os.path.join(outdir, 'weights')
    plotdir = make_plotdir(outdir)

    return (args, dgen_train, dgen_test, outdir, plotdir, cmdargs.ext)

def plot_whist(whist_train, whist_test, bins, plotdir, ext):
    """Plot inverse of the TrueE histogram"""
    f, ax = plt.subplots()
    ax.set_yscale('log')

    if whist_train is not None:
        plot_nphist1d_base(
            ax, whist_train, bins, histtype = 'step', linewidth = 2,
            label = 'Train'
        )

    plot_nphist1d_base(
        ax, whist_test, bins, histtype = 'step', linewidth = 2, label = 'Test'
    )

    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which = 'major', linestyle = 'dashed', linewidth = 1.0)
    ax.grid(True, which = 'minor', linestyle = 'dashed', linewidth = 0.5)
    ax.set_xlabel('Target')
    ax.set_ylabel('Weight')

    save_fig(
        f, os.path.join(plotdir, "weights_%s" % (whist_train is None)), ext
    )

def save_weghts(whist_train, whist_test, bins, outdir):
    """Save inverse of the TrueE histograms"""
    np.savetxt(os.path.join(outdir, "whist_train.txt"), whist_train)
    np.savetxt(os.path.join(outdir, "whist_test.txt"),  whist_test)
    np.savetxt(os.path.join(outdir, "whist_bins.txt"),  bins)

def main():
    # pylint: disable=missing-function-docstring
    args, dgen_train, dgen_test, outdir, plotdir, ext = prologue()

    whist_func = get_weights_hist_func(args)

    (_, whist_train, bins) = whist_func(dgen_train.data_loader)
    (_, whist_test,  _   ) = whist_func(dgen_test.data_loader)

    save_weghts(whist_train, whist_test, bins, outdir)

    plot_whist(None,        whist_test, bins, plotdir, ext)
    plot_whist(whist_train, whist_test, bins, plotdir, ext)

if __name__ == '__main__':
    main()


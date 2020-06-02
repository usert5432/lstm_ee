"""
Script to plot true energy distributions.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot  import (
    plot_rhist1d, plot_rhist1d_error, save_fig, remove_bottom_margin
)
from cafplot.rhist import RHist1D

from lstm_ee.eval.predict  import get_true_energies
from lstm_ee.presets       import PRESETS_EVAL
from lstm_ee.utils.eval    import standard_eval_prologue
from lstm_ee.utils.log     import setup_logging
from lstm_ee.utils.parsers import add_basic_eval_args,add_concurrency_parser

def parse_cmdargs():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("Plot targets")
    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)
    return parser.parse_args()

def plot_energy(data, weights, name, spec, log_scale = False):
    """Plot single energy distribution."""
    f, ax = plt.subplots()
    if log_scale:
        ax.set_yscale('log')

    rhist = RHist1D.from_data(data, spec.bins_x, weights)

    plot_rhist1d(
        ax, rhist,
        histtype = 'step',
        linewidth = 2,
        color     = 'C0',
        label     = "True %s. Mean: %.2e" % (
            name, np.average(data, weights = weights)
        ),
    )
    plot_rhist1d_error(
        ax, rhist, err_type = 'bar', color = 'C0', linewidth = 2
    )

    spec.decorate(ax)
    ax.legend()
    remove_bottom_margin(ax)

    return f, ax, rhist

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, _, _, outdir, _, eval_specs = standard_eval_prologue(
        cmdargs, PRESETS_EVAL
    )

    plotdir = os.path.join(outdir, 'targets')
    os.makedirs(plotdir, exist_ok = True)

    true_energies = get_true_energies(dgen)
    weights       = dgen.weights

    for label,energy in true_energies.items():
        for log_scale in [ True, False ]:
            f,_,rhist = plot_energy(
                energy, weights, eval_specs['name_map'][label],
                eval_specs['hist'][label], log_scale
            )

            save_fig(
                f, os.path.join(plotdir, '%s_log(%s)' % (label, log_scale)),
                cmdargs.ext
            )

        np.savetxt(os.path.join(plotdir, "%s_hist.txt" % (label)), rhist.hist)
        np.savetxt(os.path.join(plotdir, "%s_bins.txt" % (label)), rhist.bins)

if __name__ == '__main__':
    main()


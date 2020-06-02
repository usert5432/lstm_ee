"""Plot stats created by `gather_evals.py` script"""

import argparse
import os
import pandas as pd

from lstm_ee.presets      import PRESETS_EVAL
from lstm_ee.plot.profile import plot_profile

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser(
        "Plot stats gathered by gather_evals script"
    )

    parser.add_argument(
        'fname',
        help    = 'File with stats produceed by gather_evals script',
        metavar = 'STATS',
        type    = str,
    )

    parser.add_argument(
        '--stats',
        default = [ 'mean', 'rms', 'median', 'mu', 'sigma' ],
        dest    = 'stats',
        help    = 'Stats to plot',
        nargs   = '+',
        type    = str,
    )

    parser.add_argument(
        '-e', '--ext',
        default = 'png',
        dest    = 'ext',
        help    = 'Plot file extension',
        nargs   = '+',
        type    = str,
    )

    parser.add_argument(
        '-c', '--categorical',
        action = 'store_true',
        dest   = 'categorical',
        help   = 'Directory with saved models',
    )

    parser.add_argument(
        '-a', '--annotate',
        action = 'store_true',
        dest   = 'annotate',
        help   = 'Show y-values for each point'
    )

    parser.add_argument(
        '-s', '--sort',
        choices = [ 'x', 'y', None ],
        default = None,
        dest    = 'sort',
        help    = 'Sort values by',
        type    = str,
    )

    parser.add_argument(
        '-P', '--preset',
        help     = 'Eval preset to use',
        choices  = list(PRESETS_EVAL.keys()),
        default  = None,
        dest     = 'preset',
        type     = str,
        required = True,
    )

    parser.add_argument(
        '-b', '--baseline',
        help     = 'File with baseline statistics',
        default  = None,
        dest     = 'base_fname',
        type     = str,
    )

    return parser.parse_args()

def load_eval_stats(stats_fname):
    """Load stats produced by `gather_evals.py`"""
    return pd.read_csv(stats_fname)

def get_energy_stats(stats, energy):
    """Extract stats for a given energy type"""
    if stats is None:
        return None

    return stats[stats['energy'] == energy]

def main():
    # pylint: disable=missing-function-docstring
    cmdargs  = parse_cmdargs()
    stats    = load_eval_stats(cmdargs.fname)
    energies = stats['energy'].unique()
    label_x  = stats['gather_name'][0]
    plotdir  = "%s-plots" % (cmdargs.fname,)

    base_stats = None
    if cmdargs.base_fname is not None:
        base_stats = load_eval_stats(cmdargs.base_fname)

    os.makedirs(plotdir, exist_ok = True)

    for energy in energies:
        energy_stats      = get_energy_stats(stats, energy)
        base_energy_stats = get_energy_stats(base_stats, energy)

        for stat_name in cmdargs.stats:
            label_y = '%s : %s' % (
                stat_name, PRESETS_EVAL[cmdargs.preset]['name_map'][energy]
            )

            fname = "%s/stat_%s_%s_vs_%s_sort(%s)_cat(%s)_ann(%s)" % (
                plotdir, energy, stat_name, label_x,
                cmdargs.sort, cmdargs.categorical, cmdargs.annotate
            )

            var_list = energy_stats['gather_var']

            plot_profile(
                var_list, energy_stats, stat_name, base_energy_stats,
                label_x     = label_x,
                label_y     = label_y,
                sort_type   = cmdargs.sort,
                categorical = cmdargs.categorical,
                annotate    = cmdargs.annotate,
                fname       = fname,
                ext         = cmdargs.ext
            )


if __name__ == '__main__':
    main()


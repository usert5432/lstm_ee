"""Collect results of evals of different models into a single dataset"""

import argparse
import json
import os

import pandas as pd
from lstm_ee.args import Args

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Collect model evaluation results")

    parser.add_argument(
        '-s', '--stat-file', dest = 'stat_file', required = True, type = str,
        help = 'File path inside MODELS directories where stats are stored'
    )

    parser.add_argument(
        '-o', '--outfile', dest = 'outfile', required = True, type = str,
        help = 'File to save results to'
    )

    parser.add_argument(
        '-v', '--var', required = True, dest = 'var', nargs = '+', type = str,
        help = 'Variable that is changed between models'
    )

    parser.add_argument(
        'models', metavar = 'MODELS', nargs = '+', type = str,
        help = 'Directory with saved models'
    )

    return parser.parse_args()

def load_model_stats_list(models, stat_file, var_name):
    """Load evaluation stats for `models`"""
    stats_list = []

    if len(var_name) == 1:
        var_name = var_name[0]

    for savedir in models:
        try:
            args      = Args.load(savedir)
            stat_path = os.path.join(savedir, stat_file)
            stats     = pd.read_csv(stat_path).to_dict(orient = 'records')

            var = args[var_name]

            for s in stats:
                s['gather_var']  = json.dumps(var)
                s['gather_name'] = json.dumps(var_name)

        except IOError:
            print("Failed to load model: %s" % savedir)
            continue

        stats_list += stats

    return stats_list

def main():
    # pylint: disable=missing-function-docstring
    cmdargs    = parse_cmdargs()
    stats_list = load_model_stats_list(
        cmdargs.models, cmdargs.stat_file, cmdargs.var
    )

    pd.DataFrame(stats_list).to_csv(cmdargs.outfile, index = False)

if __name__ == '__main__':
    main()



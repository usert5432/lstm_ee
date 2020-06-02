"""Plot binned statistics plots of energy resolution vs TrueE"""

import argparse

from lstm_ee.presets         import PRESETS_EVAL
from lstm_ee.plot.hairy_mean import plot_hairy_mean_binstat
from lstm_ee.plot.binstat    import plot_binstats
from lstm_ee.utils.eval      import standard_eval_prologue
from lstm_ee.utils.log       import setup_logging
from lstm_ee.utils.parsers   import add_basic_eval_args, add_concurrency_parser
from lstm_ee.eval.predict    import (
    get_true_energies, get_base_energies, predict_energies
)

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Make Binstat plots")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    return parser.parse_args()

def make_binstat_plots(
    pred_model_dict, pred_base_dict, true_dict, weights, eval_specs,
    plotdir, ext
):
    # pylint: disable=missing-function-docstring
    plot_binstats(
        [
            (pred_base_dict,  true_dict, weights, 'Baseline', 'C0'),
            (pred_model_dict, true_dict, weights, 'Model',    'C1'),
        ],
        eval_specs['binstats_abs'], eval_specs['binstats_rel'],
        "%s/plot_aux_binstat" % (plotdir), ext
    )

    plot_hairy_mean_binstat(
        [
            (pred_base_dict,  true_dict, weights, 'Baseline', 'C0'),
            (pred_model_dict, true_dict, weights, 'Model',    'C1'),
        ],
        eval_specs['binstats_abs'], eval_specs['binstats_rel'],
        "%s/plot_aux_binstat" % (plotdir), ext
    )

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir, eval_specs = standard_eval_prologue(
        cmdargs, PRESETS_EVAL
    )

    pred_model_dict = predict_energies(args, dgen, model)
    true_dict       = get_true_energies(dgen)
    pred_base_dict  = get_base_energies(dgen, eval_specs['base_map'])

    make_binstat_plots(
        pred_model_dict, pred_base_dict, true_dict, dgen.weights,
        eval_specs, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()


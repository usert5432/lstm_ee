"""Evaluate model energy resolutions."""

import argparse

from lstm_ee.presets       import PRESETS_EVAL
from lstm_ee.utils.log     import setup_logging
from lstm_ee.utils.parsers import add_basic_eval_args, add_concurrency_parser
from lstm_ee.utils.eval    import standard_eval_prologue
from lstm_ee.eval.eval     import evaluate
from lstm_ee.plot.fom      import plot_fom

FOM_FIT_MARGIN = 0.5

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Evaluate Performance of the Model")
    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, outdir, plotdir, eval_specs = standard_eval_prologue(
        cmdargs, PRESETS_EVAL
    )

    (stats_model_dict, hCont_model_dict), (stats_base_dict, hCont_base_dict) \
        = evaluate(
            args, dgen, model, eval_specs['base_map'], eval_specs['fom'],
            FOM_FIT_MARGIN, outdir
        )

    plot_fom(
        [
            (hCont_base_dict,  stats_base_dict,  'Base ', 'left',  'C0'),
            (hCont_model_dict, stats_model_dict, 'Model', 'right', 'C1'),
        ],
        eval_specs['fom'],
        fname   = '%s/fom' % (plotdir),
        ext     = cmdargs.ext
    )

if __name__ == '__main__':
    main()


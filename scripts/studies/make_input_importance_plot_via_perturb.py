"""
Script to evaluate importance of the input variables.

It relies on adding random normal noise to the input variables and evaluating
the model performance [1]_. The idea is that if large degradation of
performance is occurred when noise is added to a given input then that input
was important.

References
----------
.. [1] J.D. Olden et al. / Ecological Modeling 178 (2004) 389-397
"""

import argparse
import logging
import os

import pandas as pd

from lstm_ee.eval.eval           import eval_model
from lstm_ee.data.data_generator import DataSmear
from lstm_ee.plot.profile        import plot_profile
from lstm_ee.presets             import PRESETS_EVAL
from lstm_ee.utils               import setup_logging
from lstm_ee.utils.eval          import standard_eval_prologue
from lstm_ee.utils.parsers       import (
    add_basic_eval_args, add_concurrency_parser
)
from lstm_ee.data.data_generator.keras_sequence import KerasSequence

def parse_cmdargs():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        "Make input importance(based on output sensitivity to random"
        " input perturbations) profiles"
    )

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    parser.add_argument(
        '--smear',
        help    = 'Smearing Value',
        default = 0.5,
        dest    = 'smear',
        type    = float,
    )

    parser.add_argument(
        '-a', '--annotate',
        action = 'store_true',
        dest   = 'annotate',
        help   = 'Show y-values for each point'
    )

    return parser.parse_args()

def prologue(cmdargs):
    """Load dataset, model and initialize output directory"""
    dgen, args, model, outdir, plotdir, eval_specs = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    plotdir    = os.path.join(
        outdir, 'input_importance_perturb_%.3e' % (cmdargs.smear)
    )
    os.makedirs(plotdir, exist_ok = True)

    return (args, model, dgen, plotdir, eval_specs)

def get_stats_for_energy(stat_list, energy):
    """Extract stats for a given energy type"""
    return pd.DataFrame([ x[energy] for x in stat_list ])

def plot_vars_profile(
    var_list, stat_list, label_x, annotate, eval_specs, plotdir, ext,
    stats_to_plot = [ 'rms', 'sigma' ],
):
    """Plot input importance vs input variable"""
    # pylint: disable=dangerous-default-value

    for energy in stat_list[0].keys():
        #for stat in stat_list[0][energy].keys():
        for stat in stats_to_plot:

            label_y = '%s(%s [%s])' % (
                stat,
                eval_specs['name_map'][energy],
                eval_specs['units_map'][energy],
            )

            fname = "%s/%s_%s_vs_%s_ann(%s)" % (
                plotdir, stat, energy, label_x, annotate
            )

            stats = get_stats_for_energy(stat_list, energy)

            plot_profile(
                var_list, stats, stat,
                base_stats  = None,
                label_x     = label_x,
                label_y     = label_y,
                sort_type   = 'y',
                annotate    = annotate,
                categorical = True,
                fname       = fname,
                ext         = ext
            )

def slice_var_generator(dgen, smear):
    """Yield IDataGeneator with smeared slice level input variable"""
    if dgen.vars_input_slice is None:
        return None

    for vname in dgen.vars_input_slice:
        dg_smear = DataSmear(
            dgen, smear = smear, affected_vars_slice = [ vname ]
        )

        yield (vname, dg_smear)

def png2d_var_generator(dgen, smear):
    """Yield IDataGeneator with smeared 2D prong level input variable"""
    if dgen.vars_input_png2d is None:
        return None

    for vname in dgen.vars_input_png2d:
        dg_smear = DataSmear(
            dgen, smear = smear, affected_vars_png2d = [ vname ]
        )

        yield (vname, dg_smear)

def png3d_var_generator(dgen, smear):
    """Yield IDataGeneator with smeared 3D prong level input variable"""
    if dgen.vars_input_png3d is None:
        return None

    for vname in dgen.vars_input_png3d:
        dg_smear = DataSmear(
            dgen, smear = smear, affected_vars_png3d = [ vname ]
        )

        yield (vname, dg_smear)

def save_stats(var_list, stat_list, label, plotdir):
    """Save input importance stats vs input variable"""
    result = []

    for idx,var in enumerate(var_list):
        for k,v in stat_list[idx].items():
            result.append({ 'var' : var, 'energy' : k, **v })

    df = pd.DataFrame.from_records(result, index = ('var', 'energy'))
    df.to_csv('%s/stats_%s.csv' % (plotdir, label))

def make_perturb_profile(
    smeared_var_generator, var_list, stat_list, args, model, eval_specs,
    plotdir, label, cmdargs
):
    """
    Evaluate performance for generators yielded by `smeared_var_generator`
    """
    if smeared_var_generator is None:
        return

    FIT_MARGIN = 0.5
    var_list   = var_list[:]
    stat_list  = stat_list[:]

    for (vname, dgen) in smeared_var_generator:
        logging.info("Evaluating '%s' var...", vname)

        stats, _ = eval_model(
            args, KerasSequence(dgen), model, eval_specs['fom'], FIT_MARGIN
        )

        var_list .append("%s : %g" % (vname, cmdargs.smear))
        stat_list.append(stats)

    plot_vars_profile(
        var_list, stat_list, label, cmdargs.annotate, eval_specs, plotdir,
        cmdargs.ext
    )

    save_stats(var_list, stat_list, label, plotdir)

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()

    cmdargs = parse_cmdargs()
    args, model, dgen, plotdir, eval_specs = prologue(cmdargs)

    var_list  = [ 'none' ]
    stat_list = [ eval_model(args, dgen, model, eval_specs['fom'])[0] ]

    make_perturb_profile(
        slice_var_generator(dgen, cmdargs.smear), var_list, stat_list,
        args, model, eval_specs, plotdir, 'slice', cmdargs
    )

    make_perturb_profile(
        png2d_var_generator(dgen, cmdargs.smear), var_list, stat_list,
        args, model, eval_specs, plotdir, 'png2d', cmdargs
    )

    make_perturb_profile(
        png3d_var_generator(dgen, cmdargs.smear), var_list, stat_list,
        args, model, eval_specs, plotdir, 'png3d', cmdargs
    )

if __name__ == '__main__':
    main()


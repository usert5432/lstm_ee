"""
Function to calculate basic evaluation metrics for the `lstm_ee` networks.
"""

import logging
import pandas as pd

from lstm_ee.eval.predict import (
    predict_energies,get_base_energies,get_true_energies
)
from lstm_ee.eval.fom   import calc_fom_stats, calc_fom_hist
from lstm_ee.eval.gauss import fit_gaussian

LOGGER = logging.getLogger('lstm_ee.eval')

def calc_fom_stats_hists(pred_dict, true_dict, weights, fom_specs, margin):
    """
    Calculate relative energy resolution stats and hists for each energy type.

    Relative energy resolution is defined as (Reco - True) / True.

    Parameters
    ----------
    pred_dict : dict
        Dictionary which keys are energy labels and values are the predicted
        values (`ndarray`, shape (N,)) of those energies.
    true_dict : dict
        Dictionary which keys are energy labels and values are the true
        values (`ndarray`, shape (N,)) of those energies. Keys of `true_dict`
        should be the same as `pred_dict`.
    weights : ndarray, shape (N,)
        An array of sample weights
    fom_specs : dict
        Dictionary where keys are energy labels and values are the `PlotSpec`
        objects that parametrize histograms of the relative energy resolution.
    margin : float
        Fraction of the height of the peak of energy resolution histogram
        (Reco - True) / True, that will be used to fit a gaussian curve.

    Returns
    -------
    stats_dict : dict
        Dictionary where keys are the energy labels (same as in `pred_dict`)
        and values are the dictionaries of values of various statistical
        properties of the relative energy resolution.

        That is `stats_dict` is a nested dictionary of the form:
        { "ENERGY_LABEL" : { "STAT_NAME" : STAT_VALUE } }

        These stats in `stats_dict` are combination of the stats calculated by
        `calc_fom_stats` and the parameters of the gaussian fit to the relative
        energy histogram..
    rhist_dict : dict
        Dictionary where keys are the energy labels (same as in `pred_dict`)
        and values are `cafplot.RHist1D` objects that contain the histograms
        of the relative energy resolution.

    See Also
    --------
    calc_fom_stats
    calc_fom_hist
    """

    stats_dict = {}
    rhist_dict = {}

    if pred_dict is None:
        return stats_dict, rhist_dict

    for k in pred_dict.keys():
        stats_dict[k] = calc_fom_stats(
            pred_dict[k], true_dict[k], weights, fom_specs[k].range_x
        )

        rhist = calc_fom_hist(
            pred_dict[k], true_dict[k], weights,
            fom_specs[k].bins_x, fom_specs[k].range_x,
        )
        rhist_dict[k] = rhist

        try:
            x = (rhist.bins_x[1:] + rhist.bins_x[:-1]) / 2
            stats_dict[k].update(fit_gaussian(x, rhist.hist, margin))
        except RuntimeError:
            LOGGER.warning("Failed to fit gaussian for: %s", k)
            stats_dict[k].update({ 'a' : 0, 'mu' : 0, 'sigma' : 0 })

    return stats_dict, rhist_dict

def eval_base(dgen, pred_map, fom_specs, margin = 0.5):
    """
    Calculate relative energy resolution stats and hists for baseline energies.

    Relative energy resolution is defined as (Reco - True) / True.

    Parameters
    ----------
    dgen : IDataGenerator
        Data generator which `dgen.data_loder` will be used to retrieve values
        of the baseline energies.
    pred_map : dict
        Dictionary that specifies mapping between energy label and a variable
        name in `dgen.data_loader` that holds baseline reconstructed energy.
    fom_specs : dict
        Dictionary where keys are energy labels and values are the `PlotSpec`
        objects that parametrize histograms of the relative energy resolution.
    margin : float
        Fraction of the height of the peak of energy resolution histogram
        (Reco - True) / True, that will be used to fit a gaussian curve.

    Returns
    -------
    (stats_dict, rhist_dict) : (dict, dict)
        Pair of dictionaries. First is a dictionary of the energy resolution
        statistics and the second is a dictionary of energy resolution
        histograms for the baseline energies.
        These values are returned by `calc_fom_stats_hists`.

    See Also
    --------
    calc_fom_stats_hists
    """

    weights   = dgen.weights
    pred_dict = get_base_energies(dgen, pred_map)
    true_dict = get_true_energies(dgen)

    return calc_fom_stats_hists(
        pred_dict, true_dict, weights, fom_specs, margin
    )

def eval_model(args, dgen, model, fom_specs, margin = 0.5):
    """
    Calculate relative energy resolution stats and hists for pred energies.

    Relative energy resolution is defined as (Reco - True) / True.

    Parameters
    ----------
    args : Args
        Arguments that define `lstm_ee` training/evaluation.
    dgen : IDataGenerator
        Data generator which will be fed to the `model` to predict energies.
    model : `keras.Model`
        Model that will be used to predict energies.
    fom_specs : dict
        Dictionary where keys are energy labels and values are the `PlotSpec`
        objects that parametrize histograms of the relative energy resolution.
    margin : float
        Fraction of the height of the peak of energy resolution histogram
        (Reco - True) / True, that will be used to fit a gaussian curve.

    Returns
    -------
    (stats_dict, rhist_dict) : (dict, dict)
        Pair of dictionaries. First is a dictionary of the energy resolution
        statistics and the second is a dictionary of energy resolution
        histograms for the energies predicted by `model`.
        These values are returned by `calc_fom_stats_hists`.

    See Also
    --------
    calc_fom_stats_hists
    """

    weights   = dgen.weights
    pred_dict = predict_energies(args, dgen, model)
    true_dict = get_true_energies(dgen)

    return calc_fom_stats_hists(
        pred_dict, true_dict, weights, fom_specs, margin
    )

def evaluate(
    args, dgen, model, base_map, fom_specs, fit_margin, outdir
):
    """Calculate relative energy resolution hists for the `model` and baseline.

    Relative energy resolution is defined as (Reco - True) / True.

    Parameters
    ----------
    args : Args
        Arguments that specify training/evaluation.
    dgen : IDataGenerator
        DataGenerator on which network will be evaluated.
    model : keras.Model
        Network to be evaluated.
    base_map : dict
        Dictionary that specifies mapping between energy label and a variable
        name in `dgen.data_loader` that holds baseline reconstructed energy.
        Possible labels are:
          [ `LABEL_PRIMARY`, `LABEL_SECONDARY`, `LABEL_TOTAL' ]
    fom_specs : dict
        Dictionary where keys are energy labels and values are the `PlotSpec`
        objects that parametrize histograms of the relative energy resolution.
    fom_spec : PlotSpec
        `PlotSpec` that parametrizes histogram of the relative energy
        resolution (Reco - True) / True.
    fit_margin : float
        Fraction of the height of the peak of energy resolution histogram
        (Reco - True) / True, that will be used to fit a gaussian curve.
    outdir : str
        Directory where evaluation statistics will be saved.

    Returns
    -------
    (stat_model_dict, rhist_model_dict) : (dict, dict)
        Pair of dictionaries. First is a dictionary of the energy resolution
        statistics and the second is a dictionary of energy resolution
        histograms for the energies predicted by `model`.
        These values are returned by `calc_fom_stats_hists`.
    (stat_base_dict, rhist_base_dict) : (dict, dict)
        Pair of dictionaries. First is a dictionary of the energy resolution
        statistics and the second is a dictionary of energy resolution
        histograms for the baseline energies.
        These values are returned by `calc_fom_stats_hists`.

    See Also
    --------
    calc_fom_stats_hists
    eval_model
    eval_base
    """
    stats_model_dict, rhist_model_dict = eval_model(
        args, dgen, model, fom_specs, fit_margin
    )
    save_model_stats(stats_model_dict, outdir)

    stats_base_dict, rhist_base_dict = eval_base(
        dgen, base_map, fom_specs, fit_margin
    )
    save_base_stats(stats_base_dict, outdir)

    return (
        (stats_model_dict, rhist_model_dict),
        (stats_base_dict , rhist_base_dict),
    )

def save_dict_as_csv(stats, fname):
    """Save dict as `pandas.DataFrame`"""
    return pd.DataFrame.from_dict(stats, orient = 'index') \
                       .to_csv(fname, index_label = "energy")

def load_csv_as_dict(fname):
    """Load `pandas.DataFrame` and convert it to a dict"""
    return pd.read_csv(fname, index_col = "energy").to_dict(orient = 'index')

def save_model_stats(stats, outdir):
    """Save stats for the energies predicted by `lstm_ee` networks."""
    save_dict_as_csv(stats, "%s/stats.csv" % (outdir))

def save_base_stats(stats, outdir):
    """Save stats for the baseline energies."""
    save_dict_as_csv(stats, "%s/stats_base.csv" % (outdir))

def load_model_stats(outdir):
    """Load stats for the energies predicted by `lstm_ee` networks."""
    return load_csv_as_dict("%s/stats.csv" % (outdir))

def load_base_stats(outdir):
    """Load stats for the baseline energies."""
    return load_csv_as_dict("%s/stats_base.csv" % (outdir))



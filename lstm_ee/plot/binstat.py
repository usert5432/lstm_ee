"""
Functions to make binstat plots.
"""

import matplotlib.pyplot as plt

from cafplot.plot import plot_nphist1d_base, save_fig
from lstm_ee.eval.binned_stats import calc_binned_stats, calc_stat

def plot_binstat_single(ax, x, y, weights, label, color, spec, stat):
    """Add a plot of single binstat to axes `ax`"""

    binstats = calc_binned_stats(x, y, weights, spec.bins_x, stat)
    fullstat = calc_stat(y, weights, stat)

    plot_nphist1d_base(
        ax, binstats, spec.bins_x,
        color     = color,
        label     = '%s : %.3e' % (label, fullstat),
        linewidth = 2
    )

    return binstats

def plot_binstat_base(
    list_of_pred_true_weight_label_color, key, spec, stat, is_rel = False
):
    """Plot binstats of relative energy resolution vs true energy."""
    spec = spec.copy()

    if spec.title is None:
        spec.title = stat.upper()
    else:
        spec.title = "%s( %s )" % (stat.upper(), spec.title)

    f, ax = plt.subplots()

    for pred,true,weights,label,color in list_of_pred_true_weight_label_color:
        x = true[key]
        y = (pred[key] - true[key])

        if is_rel:
            y = y / x

        plot_binstat_single(ax, x, y, weights, label, color, spec, stat)

    ax.axhline(0, 0, 1, color = 'C2', linestyle = 'dashed')
    spec.decorate(ax)

    ax.legend()

    return f, ax

def plot_binstats(
    list_of_pred_true_weight_label_color,
    plot_specs_abs, plot_specs_rel, fname, ext,
    stat_list = [ 'mean', 'rms' ]
):
    """Make and save binstat plots of energy resolution vs true energy.

    Parameters
    ----------
    list_of_pred_true_weight_label_color : list
        List of tuples of the form (pred, true, weights, label, color) where:
        pred : dict
            Dictionary where keys are energy labels and values are the
            `ndarray` (shape (N,)) of predicted energies.
        true : `ndarray`, shape (N,)
            Dictionary where keys are energy labels and values are the
            `ndarray` (shape (N,)) of true energies.
        weights : `ndarray`, shape (N,)
            Sample weights.
        label : str
            Plot label.
        color : str
            Line color.
        A separate plot will be made for each key in the `pred`.
        Lines for all elements of the `list_of_pred_true_weight_label_color`
        will be drawn on each plot.
    plot_specs_abs : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify axes and bins of the absolute energy resolution plots.
    plot_specs_rel : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify axes and bins of the relative energy resolution plots.
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    stat_list : list, optional
        List of statistic properties for which binstat plots will be made.
        Default: [ 'mean', 'rms' ]
    """
    # pylint: disable=dangerous-default-value

    plot_types = plot_specs_abs.keys()

    for is_rel,spec,rel_label in zip(
        [ True,           False ],
        [ plot_specs_rel, plot_specs_abs ],
        [ 'rel',          'abs' ]
    ):
        for k in plot_types:
            for stat in stat_list:
                f, _ = plot_binstat_base(
                    list_of_pred_true_weight_label_color,
                    k, spec[k], stat, is_rel
                )

                fullname = "%s_%s_%s_%s" % (fname, k, stat, rel_label)
                save_fig(f, fullname, ext)
                plt.close(f)


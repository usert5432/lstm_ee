"""
Functions to make binstat plots of means with error bars.
"""

import matplotlib.pyplot as plt

from lstm_ee.eval.binned_stats import calc_binned_stats, calc_stat
from cafplot.plot.nphist import plot_nphist1d_base, plot_nphist1d_error
from cafplot.plot import save_fig

def plot_hairy_mean_binstat_single(ax, x, y, weights, bins, color, label, err):
    """Add binstat plot of mean with error bars to axes `ax`"""

    mean_binstats = calc_binned_stats(x, y, weights, bins, 'mean')
    mean_fullstat = calc_stat(y, weights, 'mean')

    err_binstats = calc_binned_stats(x, y, weights, bins, err)
    err_fullstat = calc_stat(y, weights, err)

    plot_nphist1d_base(
        ax, mean_binstats, bins,
        color     = color,
        label     = '%s : MEAN: %.3e, %s: %.3e' % (
            label, mean_fullstat, err.upper(), err_fullstat
        ),
        linewidth = 2,
    )

    plot_nphist1d_error(
        ax, mean_binstats - err_binstats, mean_binstats + err_binstats,
        bins, err_type = 'bar', linewidth = 2, color = color, alpha = 0.7
    )

def plot_hairy_mean_binstat_base(
    list_of_pred_true_weight_label_color, key, spec,
    is_rel = False, err = 'rms'
):
    """Plot binstats of means of relative energy resolution vs true energy."""
    spec = spec.copy()

    if spec.title is None:
        spec.title = 'MEAN + E[ %s ]' % (err.upper())
    else:
        spec.title = '(MEAN + E[ %s ]) ( %s )' % (err.upper(), spec.title)

    f, ax = plt.subplots()

    for pred,true,weights,label,color in list_of_pred_true_weight_label_color:
        x = true[key]
        y = (pred[key] - true[key])

        if is_rel:
            y = y / x

        plot_hairy_mean_binstat_single(
            ax, x, y, weights, spec.bins_x, color, label, err
        )


    ax.axhline(0, 0, 1, color = 'C2', linestyle = 'dashed')
    spec.decorate(ax)

    ax.legend()

    return f, ax

def plot_hairy_mean_binstat(
    list_of_pred_true_weight_label_color,
    plot_specs_abs, plot_specs_rel, fname, ext,
    err_list = [ 'rms', 'stderr' ]
):

    """
    Make and save binstat plots of means of energy resolution vs true energy.

    This function makes binned statistics plots of means of the relative
    energy resolution vs true energy and adds error bars to the means.

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
    err_list : list, optional
        List of statistic properties which values will be used for determining
        errors of the binstat plot.
        Default: [ 'mean', 'rms' ]

    See Also
    --------
    plot_binstats
    """
    # pylint: disable=dangerous-default-value

    plot_types = plot_specs_abs.keys()

    for is_rel,spec,rel_label in zip(
        [ True,           False ],
        [ plot_specs_rel, plot_specs_abs ],
        [ 'rel',          'abs' ]
    ):
        for k in plot_types:
            for err in err_list:
                f, _ = plot_hairy_mean_binstat_base(
                    list_of_pred_true_weight_label_color,
                    k, spec[k], is_rel = is_rel, err = err
                )

                fullname = "%s_%s_hairy_mean-%s_%s" % (
                    fname, k, err, rel_label
                )
                save_fig(f, fullname, ext)
                plt.close(f)


"""
Functions to make plots of energy histograms.
"""

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot  import (
    make_figure_with_ratio,
    plot_rhist1d, plot_rhist1d_error, plot_rhist1d_ratios,
    save_fig, remove_bottom_margin
)
from cafplot.rhist import RHist1D

def plot_hist_base(
    list_of_data_weight_label_color, key, spec, ratio_plot_type, stat_err,
    log = False
):
    """Plot multiple energy histograms"""

    if ratio_plot_type is not None:
        f, ax, axr = make_figure_with_ratio()
    else:
        f, ax = plt.subplots()

    if log:
        ax.set_yscale('log')

    list_of_rhist_color = []

    for (data,weights,label,color) in list_of_data_weight_label_color:
        rhist = RHist1D.from_data(data[key], spec.bins_x, weights)

        centers = (rhist.bins[1:] + rhist.bins[:-1]) / 2
        mean = np.average(centers, weights = rhist.hist)

        plot_rhist1d(
            ax, rhist,
            histtype = 'step',
            marker    = None,
            linestyle = '-',
            linewidth = 2,
            label     = "%s. MEAN = %.3e" % (label, mean),
            color     = color,
        )

        if stat_err:
            plot_rhist1d_error(
                ax, rhist, err_type = 'bar', color = color, linewidth = 2,
                alpha = 0.8
            )

        list_of_rhist_color.append((rhist, color))

    spec.decorate(ax, ratio_plot_type)

    if not log:
        remove_bottom_margin(ax)

    ax.legend()

    if ratio_plot_type is not None:
        plot_rhist1d_ratios(
            axr,
            [rhist_color[0] for rhist_color in list_of_rhist_color],
            [rhist_color[1] for rhist_color in list_of_rhist_color],
            err_kwargs = { 'err_type' : 'bar' if stat_err else None },
        )
        spec.decorate_ratio(axr, ratio_plot_type)

    return f, ax

def plot_energy_hists(
    list_of_data_weight_label_color, plot_specs, fname, ext, log = False
):
    """Make and save plots of energy histograms.

    Parameters
    ----------
    list_of_pred_true_weight_label_color : list
        List of tuples of the form (pred, true, weights, label, color) where:
        data : dict
            Dictionary where keys are energy labels and values are the
            `ndarray` (shape (N,)) of energies.
        weights : `ndarray`, shape (N,)
            Sample weights.
        label : str
            Plot label.
        color : str
            Line color.
        A separate plot will be made for each key in the `data`.
        Lines for all elements of the `list_of_data_weight_label_color`
        will be drawn on each plot.
    plot_specs : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify axes and bins of the energy plots.
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    log : bool
        If True then the vertical axis will have logarithmic scale.
        Default: False.
    """

    for k in plot_specs.keys():
        for ratio_plot_type in [ None, 'auto', 'fixed' ]:
            for stat_err in [ True, False ]:

                f, _ = plot_hist_base(
                    list_of_data_weight_label_color, k, plot_specs[k],
                    ratio_plot_type, stat_err, log
                )

                fullname = "%s_%s_ratio-%s_staterr-%s" % (
                    fname, k, ratio_plot_type, stat_err
                )
                save_fig(f, fullname, ext)
                plt.close(f)


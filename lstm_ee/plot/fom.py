"""
Functions to make plots of relative energy resolution histograms.
"""

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot import plot_rhist1d, save_fig, remove_bottom_margin
from lstm_ee.eval.gauss import gaussian

def determine_alignment(pos):
    """Parse position of the textbox"""

    tokens = pos.split('-')
    if len(tokens) > 2:
        raise RuntimeError("Unknown position: %s" % (pos))

    vpos = None
    hpos = None

    for token in tokens:
        if token in [ 'right', 'left' ]:
            hpos = token
        elif token in [ 'top', 'bottom' ]:
            vpos = token
        else:
            raise RuntimeError("Unknown position: %s" % (token))

    return (vpos, hpos)

def get_corner_coord(vhpos):
    """Convert textbox corner coordinate from str to float"""
    if (vhpos is None) or (vhpos == 'right') or (vhpos == 'top'):
        return 0.9
    else:
        return 0.1

def add_stats_text(ax, stats, label, pos, **kwargs):
    """Draw textbox with statistical information"""

    text = label
    text += "\n\nStats:\nMean %.3e\nRMS %.3e\nStd %.3e"     \
          % (stats['mean'], stats['rms'], stats['stdev'])

    text += "\n\nGauss:\nMu %.3e\nSigma %.3e\nPeak %.3e"    \
          % (stats['mu'], stats['sigma'], stats['a'])

    vpos, hpos = determine_alignment(pos)

    x = get_corner_coord(hpos)
    y = get_corner_coord(vpos)

    ax.text(
        x, y, text,
        horizontalalignment = hpos or 'right',
        verticalalignment   = vpos or 'top',
        transform           = ax.transAxes,
        **kwargs
    )

def plot_gauss_fit(ax, bins, stats, **kwargs):
    """Plot a gaussian fit"""
    x = np.linspace(bins[0], bins[-1], 1000)
    y = gaussian(x, stats['a'], stats['mu'], stats['sigma'])

    ax.plot(x, y, **kwargs)

def plot_fom_base(ax, rhist, stats, label, pos, color, spec):
    """
    Plot relative energy resolution histogram and draw textbox with stat info.
    """

    plot_rhist1d(ax, rhist, histtype = 'step', label = label, color = color)
    plot_gauss_fit(ax, rhist.bins_x, stats, color = color)

    add_stats_text(ax, stats, label, pos, color = color)
    spec.decorate(ax)

def plot_fom(list_of_rhist_stats_labels_pos_colors, plot_specs, fname, ext):
    """Make and save plots of relative energy resolution histograms

    Parameters
    ----------
    list_of_rhist_stats_labels_pos_colors : list
        List of tuples of the form (rhist, stats, label, pos, color) where:
        rhist : dict
            Dictionary where keys are energy labels and the values are
            `cafplot.RHist1D` containing relative energy resolution histograms.
            C.f.  `calc_fom_stats_hists`.
        stats : dict
            Dictionary where keys are energy labels and values are the
            dictionaries of values of various statistical properties of the
            relative energy resolution.
            C.f. `calc_fom_stats_hists`.
        label : str
            Plot label.
        pos : str
            Position of the textbox with statistical information.
            It should be of the form "([top|bottom]-)?[left|right]".
        color : str
            Line color.
        A separate plot will be made for each key in the `pred`.
        Lines for all elements of the `list_of_pred_true_weight_label_color`
        will be drawn on each plot.
    plot_specs : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify axes and bins of the relative energy resolution plots.
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    """

    for k,spec in plot_specs.items():

        f, ax = plt.subplots()

        for (rhist_dict,stats_dict,label,pos,color) in \
                list_of_rhist_stats_labels_pos_colors:

            plot_fom_base(
                ax, rhist_dict[k], stats_dict[k], label, pos, color, spec
            )

        remove_bottom_margin(ax)

        save_fig(f, '%s_%s' % (fname, k), ext)
        plt.close(f)


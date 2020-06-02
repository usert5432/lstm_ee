"""
Functions to make auxiliary (i.e. mostly useless) evaluation plots.
"""

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from cafplot.plot      import save_fig

def plot_rel_res_vs_true_base(pred, true, weights, spec, logNorm = False):
    """Plot 2D histogram of relative energy resolution vs true energy"""

    f, ax = plt.subplots()

    res = (pred - true) / true

    if logNorm:
        kwargs = { 'norm' : LogNorm() }
    else:
        kwargs = { }

    _hist2d, _, _, image = ax.hist2d(
        true, res,
        bins    = [spec.bins_x, spec.bins_y],
        weights = weights,
        cmin = 1e-5,
        **kwargs
    )

    ax.axhline(0, 0, 1, color = 'C2', linestyle = 'dashed')

    spec.decorate(ax)
    f.colorbar(image)

    return f, ax

def plot_rel_res_vs_true(
    pred_dict, true_dict, weights, plot_specs, fname, ext
):
    """
    Make and save 2D hist plots of relative energy resolution vs true energy.

    Parameters
    ----------
    pred_dict : dict
        Dictionary where keys are energy labels and values are `ndarray`
        (shape (N,)) of predicted energies.
    true_dict : dict
        Dictionary where keys are energy labels and values are `ndarray`
        (shape (N,)) of true energies.
    weights : ndarray, shape (N,)
        Sample weights.
    plot_specs : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify the plot style and histogram bins.
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    """

    for k in pred_dict.keys():
        pred = pred_dict[k]
        true = true_dict[k]
        spec = plot_specs[k]

        for logNorm in [True, False]:
            try:
                f, _ = plot_rel_res_vs_true_base(
                    pred, true, weights, spec, logNorm
                )
            except ValueError as e:
                print("Failed to make plot: %s" % (str(e)))
                continue

            path = "%s_%s_log(%s)" % (fname, k, logNorm)

            save_fig(f, path, ext)
            plt.close(f)


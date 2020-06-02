"""
Functions to make plots of training history.
"""

import matplotlib.pyplot as plt
import scipy.signal

from cafplot.plot import save_fig

def plot_scatter_with_average(
    ax, x, y_df, color, label, marker_size, smooth = False
):
    """Add a scatter plot of training metric to axes `ax`"""

    ax.scatter(x, y_df.values, color = color, label = label, s = marker_size)

    if smooth:
        y_avr = scipy.signal.savgol_filter(y_df.values.ravel(), 5, 2)
    else:
        y_avr = y_df.rolling(window = 5, center = True, min_periods = 1)    \
                    .mean().values

    ax.plot(x, y_avr, color = color, linestyle = 'dashed', linewidth = 1)

def plot_train_val_history(x_name, y_name, log, skip, plotdir, ext):
    """Make and save plot of the training metric for train/validation samples

    Parameters
    ----------
    x_name ; str
        Label in the `log` which values will be used as x axis.
    y_name ; str
        Label in the `log` which values will be used as y axis.
        It is assumed that metrics evaluated on the training dataset
        will have name `y_name` and metrics evaluated on the validation
        dataset will have name "val_" + `y_name".
    log : pandas.DataFrame
        `DataFrame` that holds training history.
    skip : int
        Number of initial data point to skip when making plot.
    plotdir : str
        Directory where plot will be saved.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    """

    x = log.loc[skip:, x_name].values.ravel()

    y_train = log.loc[skip:, y_name]
    y_val   = log.loc[skip:, 'val_' + y_name]

    f, ax = plt.subplots()

    plot_scatter_with_average(ax, x, y_train, 'C0', 'Train', 10)
    plot_scatter_with_average(ax, x, y_val,   'C1', 'Test',  10)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('%s vs %s' % (y_name, x_name))

    ax.legend()

    fname = "%s/plot_%s_vs_%s_%d" % (plotdir, y_name, x_name, skip)
    save_fig(f, fname, ext)


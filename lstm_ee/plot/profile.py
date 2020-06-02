"""
Functions to make plots of evaluation metrics vs training config parameter(-s).
"""

import numbers
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot       import save_fig
from lstm_ee.eval.stats import get_stat_err


def plot_profile_base(
    x, y, err, label_x, label_y, annotate, categorical, scale_x, scale_y
):
    """Make a scatter plot of eval metric `y` vs training config param `x`"""
    f, ax = plt.subplots()

    if scale_y is not None:
        ax.set_yscale(scale_y)

    if scale_x is not None:
        ax.set_xscale(scale_x)

    ax.scatter(x, y, color = 'C0', label = 'model')

    if err is not None:
        ax.vlines(x, y - err, y + err, color = 'C0')

    if (label_x is not None) and (label_y is not None):
        ax.set_title('%s vs %s' % (label_y, label_x))

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    if annotate:
        # pylint: disable=consider-using-enumerate
        for i in range(len(x)):
            ax.annotate(
                "%.1e" % y[i],
                [x[i], y[i]],
                horizontalalignment = 'center',
                verticalalignment   = 'bottom',
                rotation            =  45,
                fontsize            = 'smaller'
            )

    if categorical:
        x = [ "\n".join(wrap(l, 70)) for l in x ]
        ax.set_xticklabels(x)
        f.autofmt_xdate(rotation = 45)

    ax.legend()

    return f, ax

def get_err(dict_of_stats, stat):
    """Estimate errors for a statistical property `stat`"""
    result = get_stat_err(dict_of_stats, stat)

    if result is None:
        return None

    return np.array(result)

def sort_data(x, y, err, sort_type):
    """Simultaneously sort (x, y, err) according to `sort_type`"""

    if sort_type is None:
        return (x, y, err)

    if sort_type == 'x':
        sorted_indices = np.argsort(x)
    elif sort_type == 'y':
        sorted_indices = np.argsort(np.abs(y))
    else:
        raise ValueError("Unknown sort type: %s" % (sort_type))

    return (
        x[sorted_indices], y[sorted_indices],
        None if (err is None) else err[sorted_indices]
    )

def prepare_x_var(var_list, categorical):
    """Pack values of training config parameter `x` into a `numpy` `ndarray`"""
    if categorical:
        return np.array([str(z) for z in var_list])
    else:
        x = [z if z is not None else np.nan for z in var_list]
        return np.array(x)

def get_x_scales(x, categorical):
    """Get possible types of `x` axis scales for a given training param"""

    if (
           categorical
        or np.any([not isinstance(z, numbers.Number) for z in x])
        or np.any(np.isnan(x))
    ):
        return [ None ]
    else:
        return [ 'linear', 'log' ]

def plot_baseline_stats(ax, base_stats, stat_name):
    """Plot value of the eval metric `stat_name` for the baseline energies

    Parameters
    ----------
    ax : Axes
        Matplotlib axes on which baseline metric will be plotted.
    base_stats : dict or None
        Dictionary where keys are the names of evaluation metrics and values
        are `ndarray` of corresponding values. The `ndarray`s are assumed to
        contain just multiple copies of the same value, therefore only the
        first element of `ndarray` will be used for plotting.
        If None, then this function will not plot anything.
    stat_name : str
        Name of the evaluation metric in `base_stats` that will be plotted.
    """
    if base_stats is None:
        return

    stat_value = base_stats[stat_name]
    if len(stat_value) != 1:
        raise RuntimeError(
            "Incorrect number of baseline stats: %s" % (stat_value)
        )

    stat_value = stat_value.values[0]

    ax.axhline(
        stat_value, 0, 1, color = 'C3', label = 'Baseline',
        linestyle = 'dashed'
    )

def plot_profile(
    var_list, stats, stat_name,
    base_stats  = None,
    label_x     = None,
    label_y     = None,
    sort_type   = None,
    annotate    = False,
    categorical = True,
    fname       = None,
    ext         = 'png'
):
    """
    Make and save plots of evaluation metric vs training config parameter(-s).

    This function will make and save a number plots (one for different
    axis scales) of the evaluation metric `stat_name` vs values of the
    training config parameter in `var_list` for different models.

    Parameters
    ----------
    var_list : list
        List of values of the configuration parameters. Each element in
        `var_list` specifies different training. Value of the configuration
        parameter can be of any type. These values will be used for the x axis.
    stats : dict
        Dictionary where keys are the names of evaluation metrics and the
        values are the `ndarray` of evaluation metric. Each `ndarray` should
        have length equal to len(`var_list`) and elements of the `ndarray`
        correspond to the elements of `var_list`.
    stat_name : str
        Name of the evaluation metric in `stats` that will be used as y
        coordinate when making a plot.
    base_stats : dict or None, optional
        Dictionary where keys are the names of evaluation metrics and the
        values are the `ndarray` of evaluation metric for the baseline
        energies. Each `ndarray` is assumed to have length of `var_list`, but
        contain copies of the same value (baseline energy is independent
        of training). Therefore, only the first element of each `ndarray` will
        be used when plotting baseline metrics.
        If None, then the values of baseline metrics will not be shown.
        Default: None.
    label_x : str or None, optional
        Label of the x axis. Default: None.
    label_y : str or None, optional
        Label of the y axis. Default: None.
    sort_type : { 'x', 'y', None }, optional
        If not None, then the points will be ordered on the x axis based on
        their values in the axis specified at `sort_type`.
        For example, if the configuration parameter is a categorical variable
        (e.g. model name) then the x axis won't have any natural order, and
        it may make sense to points by their y coordinates.
        Default: None.
    annotate : bool, optional
        If True, then y value will be shown b next to each data point.
        Default: False.
    categorical : bool, optional
        Whether to assume that the x variable is a categorical (as opposed to
        numerical). For example, if `var_list` contains values of the learning
        rate, then it is a numerical variable. On the other hand if `var_list`
        contains names of the models (str), then such variable cannot be
        represented as a number and therefore categorical.
        If x variable is categorical then it does not make sense to plot it
        with logarithmic scale or convert values to numbers. This parameter
        hints `plot_profile_base` not to do those things.
        Default: True
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    """
    x   = prepare_x_var(var_list, categorical)
    y   = stats[stat_name].values.ravel()
    err = get_err(stats, stat_name)

    x, y, err = sort_data(x, y, err, sort_type)

    scale_x_list = get_x_scales(x, categorical)
    scale_y_list = [ 'linear', 'symlog' if (np.any(y <= 0)) else 'log' ]

    for scale_y in scale_y_list:
        for scale_x in scale_x_list:
            f, ax = plot_profile_base(
                x, y, err, label_x, label_y, annotate, categorical,
                scale_x, scale_y
            )
            plot_baseline_stats(ax, base_stats, stat_name)

            fullname = "%s_xs(%s)_ys(%s)" % (fname, scale_x, scale_y)
            save_fig(f, fullname, ext)
            plt.close(f)


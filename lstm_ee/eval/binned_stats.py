"""
Functions for calculating binned statistics.
"""

import numpy as np
from .stats import calc_stat

def calc_binned_stats(x, y, weights, bins_x, stat = 'mean'):
    """Calculate binned statistics for the data.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Coordinates of data points that will be used to determine the bin
        indices.
    y : ndarray, shape (N,)
        Coordinates of data points that will be used to calculate statistics
        for each bin.
    weights : ndarray, shape (N,)
        Weight of each data point.
    bins_x : list of float
        List of bin edges.
    stat : { 'mean', 'rms', 'stdev', 'stderr', 'median' }
        Name of the binned statistics to calculate.

    Returns
    -------
    ndarray, shape (len(bins_x) - 1,)
        Value of statistics `stat` for each bin defined by `bins_x`.
    """

    bin_idx = np.digitize(x, bins = bins_x)

    results = []

    for i in range(1, len(bins_x)):
        cur_y = y[bin_idx == i]
        cur_w = weights[bin_idx == i]

        value = calc_stat(cur_y, cur_w, stat)

        results.append(value)

    return np.array(results)


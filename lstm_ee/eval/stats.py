"""
Functions to calculate various statistical properties.
"""

import numpy as np

def calc_median(v, w):
    """Calculate median of a weighted dataset.

    Parameters
    ----------
    v : ndarray, shape (N,)
        Array of values for which median will be found.
    w : ndarray, shape (N,)
        Array of weights.

    Returns
    -------
    float
        Median of the `v` array.
    """
    # pylint: disable=len-as-condition
    if len(v):
        return 0

    sorted_indices = np.argsort(v)

    v = v[sorted_indices]
    w = w[sorted_indices]

    cumsum     = np.cumsum(w)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)

    return v[median_idx]

def calc_stderr(v, w):
    """Calculate standard error of a weighted dataset.

    This is the unbiased version of the standard error as defined in
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    Parameters
    ----------
    v : ndarray, shape (N,)
        Array of values for which standard error will be found.
    w : ndarray, shape (N,)
        Array of weights.

    Returns
    -------
    float
        Standard error of the `v` array.
    """

    w = w / sum(w)

    mean   = np.sum(w * v)
    result = np.sum(w**2 * (v - mean)**2)

    if len(v) > 1:
        result = len(v) / (len(v) - 1) * result

    return np.sqrt(result)

def calc_all_stats(data, weights):
    """Calculate multiple statistics for a weighted dataset.

    Parameters
    ----------
    data : ndarray, shape (N,)
        Array of values.
    w : ndarray, shape (N,)
        Array of weights.

    Returns
    -------
    dict
        A dictionary where labels are the names of statistical properties
        and the values are the values of corresponding statistical properties.

    Notes
    -----
    This function calculates the following statistical properties of `data`:
        - "mean"   -- average value
        - "rms"    -- root mean squared value
        - "stdev"  -- standard deviation
        - "stderr" -- unbiased standard error
        - "median" -- median
        - "m1"     -- First moment, same as "mean"
        - "m2"     -- Second moment, same as "rms"**2
        - "m4"     -- Fourth moment
        - "x1var"  -- Biased variance of `data`.
        - "x2var"  -- Biased variance of `data**2`
        - "m1var"  -- Biased variance of the first moment. Aka biased "stderr".
        - "m2var"  -- Biased variance of the second moment.

    I have derived "[mx]1var" and "[mx]2var" myself. Need to find some credible
    reference and verify values. And replace them by the unbiased versions.
    """

    weights = weights / sum(weights)

    m1 = np.sum(weights * data)
    m2 = np.sum(weights * data**2)
    m4 = np.sum(weights * data**4)

    # Variances of x and x^2
    # TODO: find unbiased versions
    x1var = m2 - m1**2
    x2var = m4 - m2**2

    # Variances of m1 and m2
    m1var = np.sum(weights**2) * x1var
    m2var = np.sum(weights**2) * x2var

    mean   = m1
    rms    = np.sqrt(m2)
    stdev  = np.sqrt(x1var)
    stderr = np.sqrt(m1var)
    median = calc_median(data, weights)

    return {
        'mean'   : mean,
        'rms'    : rms,
        'stdev'  : stdev,
        'stderr' : stderr,
        'median' : median,
        'm1'     : m1,
        'm2'     : m2,
        'm4'     : m4,
        'x1var'  : x1var,
        'x2var'  : x2var,
        'm1var'  : m1var,
        'm2var'  : m2var,
    }

def calc_stat(v, w, stat):
    """Calculate statistical property of the weighted dataset.

    Parameters
    ----------
    data : ndarray, shape (N,)
        Array of values.
    w : ndarray, shape (N,)
        Array of weights.
    stat : { 'mean', 'rms', 'stdev', 'stderr', 'median' }
        Type of the statistical property to calculate.

    Returns
    -------
    float
        Value of the statistical property `stat`.
    """
    # pylint: disable=len-as-condition
    if len(v) == 0:
        return np.nan

    w = w / sum(w)

    if stat == 'mean':
        return np.sum(w * v)

    elif stat == 'rms':
        return np.sqrt(np.sum(w * v**2))

    elif stat == 'std':
        return np.sqrt(calc_stat(v, w, 'rms')**2 - calc_stat(v, w, 'mean')**2)

    elif stat == 'stderr':
        return calc_stderr(v, w)

    elif stat == 'median':
        return calc_median(v, w)

    else:
        raise ValueError("Unknown stat: %s" % stat)

def get_stat_err(stats_dict, stat):
    """Get estimate of error of the statistical property `stat`.

    Parameters
    ----------
    stats_dict : dict
        Dictionary of statistical properties as returned by `calc_all_stats`.
    stat : { 'mean', 'rms' } or str
        Name of the statistical property which error will be estimated.
        If some generic str than this function will return None.

    Returns
    -------
    float or None
        Estimated error of the statistical property `stat`.

    Notes
    -----
    I have derived error of the "rms" myself. Need to find some credible
    reference and verify it. Also this error is biased, need to find unbiased
    version.
    """

    if stat == 'mean':
        return stats_dict['stderr']

    # NOTE: about error calculation
    #    rms      = sqrt(E[x^2]) == sqrt(m2)
    #    err(rms) = err(m2) * (D[sqrt(m2)]/D[m2])
    #             = m2err * 1/(2 * sqrt(m2))

    if stat == 'rms':
        m2    = stats_dict['m2']
        m2err = np.sqrt(stats_dict['m2var'])

        return (1/2) * (m2err / np.sqrt(m2))

    return None


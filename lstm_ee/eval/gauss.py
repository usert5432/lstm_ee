"""
Functions to fit gaussian profile to the energy resolution histograms.
"""

import numpy as np
import scipy
import scipy.optimize

def gaussian(x, a, x0, sigma):
    """Gaussian-like curve"""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussian_diff(x, a, x0, sigma):
    """Derivative of the Gaussian-like curve"""

    exp = np.exp(-(x - x0)**2 / (2 * sigma**2))
    f   = a * exp

    diff_a      = exp
    diff_x0     = (x - x0) / (sigma**2) * f
    diff_sigma  = (x - x0)**2 / (sigma**3) * f

    return np.vstack([diff_a, diff_x0, diff_sigma]).T

def drop_outliers(x, hist, threshold, margin = 0):
    """Remove outliers from a histogram.

    This function tries to remove outliers of the histogram `hist`.
    Outlier defined is a bin where values are less than `threshold`.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Bin centers of the histogram.
    hist : ndarray, shape (N,)
        Values of the histogram.
    threshold : float
        Threshold that will be used to judge whether given point is an outlier
        or not.
    margin : int, optional
        Number of bins around the boundary of the non-outliers that won't be
        dropped. Default: 0.

    Returns
    -------
    x : ndarray, shape(M,)
        Bin centers of the histogram without outliers.
    hist : ndarray, shape(M,)
        Values of the histogram without outliers.
    """
    is_above = (hist >= threshold)

    indices = np.arange(0, len(hist))
    indices_above = indices[is_above]

    imin = indices_above[0]
    imax = indices_above[-1]

    imin = max(0, imin - margin)
    imax = min(len(x), imax + margin)

    return (x[imin:imax], hist[imin:imax])

def fit_gaussian(x, hist, drop_margin = 0.5, min_bins = 1):
    """Fit gaussian curve to a histogram.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Bin centers of the histogram.
    hist : ndarray, shape (N,)
        Values of the histogram.
    drop_margin : float, optional
        This value will be used to determine part of the histogram for which
        gaussian fit will be performed. The part is determined by finding
        the peak value of the `hist` and taking point that are above
        peak value of `hist` times `drop_margin`.
        In other words, if `drop_margin` = 0.5, then the gaussian will be fit
        only on the top half of the histogram.
        Default: 0.5
    min_bins : int, optional
        Minimum number of non-empty bins that are required to perform the fit.
        If number of non-empty bins in `hist` is less than `min_bins` then
        an exception will be raised.

    Returns
    -------
    dict
        Parameters of the gaussian fit `gaussian`:

    Raises
    ------
    RuntimeError
        If the number of non-empty bins in `hist` is less than `min_bins`.
        Or if fit fails for some reason.
    """

    peak    = np.max(hist)
    x, hist = drop_outliers(x, hist, peak * drop_margin)

    if np.count_nonzero(hist) < min_bins:
        raise RuntimeError("Too few nonempty bins")

    # This is because of possible numerical errors in the weighted numpy funcs.
    weights = hist / sum(hist)

    guess_a     = np.max(hist)
    guess_mu    = np.sum(weights * x)
    guess_sigma = np.sqrt(np.sum(weights * x**2) - guess_mu**2)

    guess = [guess_a, guess_mu, guess_sigma]

    popt, _pcov = scipy.optimize.curve_fit(
        gaussian,
        xdata    = x,
        ydata    = hist,
        p0       = guess,
        bounds   = [
            [0,      x[0],  0],
            [np.inf, x[-1], np.inf]
        ],
        jac      = gaussian_diff,
        max_nfev = 10000
    )

    return { 'a' : popt[0], 'mu' : popt[1], 'sigma' : popt[2] }


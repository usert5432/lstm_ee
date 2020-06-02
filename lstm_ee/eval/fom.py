"""
Functions to calculate stats/hists of the relative energy resolution.

The relative energy resolution is defined as (Reco - True) / True.
"""

from cafplot.rhist import RHist1D
from .stats import calc_all_stats

def calc_fom_hist(pred, true, weights, bins, range):
    """Calculate relative energy resolution histogram.

    Parameters
    ----------
    pred : ndarray, shape (N,)
        Array of predicted energies.
    true : ndarray, shape (N,)
        Array of true energies.
    weights : ndarray, shape (N,)
        Array of sample weights.
    bins : int or ndarray
        If int then defined the number of bins in a histogram.
        If ndarray then `bins` defines edges of the histogram.
        C.f. `np.histogram`
    range : (float, float) or None
        Range of a histogram (lower, upper). If None, range will be determined
        according to the `np.histogram` rules.

    Returns
    -------
    cafplot.RHist1D
        `Rhist1D` object containing the relative energy resolution histogram.
    """
    # pylint: disable=redefined-builtin
    fom  = (pred - true) / true
    return RHist1D.from_data(fom, bins, weights, range)

def calc_fom_stats(pred, true, weights, range = (-1, 1)):
    """Calculate relative energy resolution statistics.

    Parameters
    ----------
    pred : ndarray, shape (N,)
        Array of predicted energies.
    true : ndarray, shape (N,)
        Array of true energies.
    weights : ndarray, shape (N,)
        Array of sample weights.
    range : (float, float)
        Range of energy resolution values that will be used in calculation
        of statistics. Useful to remove outliers.

    Returns
    -------
    dict
        Dictionary where keys are the names of various statistics and values
        are the values of corresponding stats.

    See Also
    --------
    calc_all_stats
    """

    # pylint: disable=redefined-builtin

    fom  = (pred - true) / true
    mask = ((fom > range[0]) & (fom < range[1]))

    return calc_all_stats(fom[mask], weights[mask])


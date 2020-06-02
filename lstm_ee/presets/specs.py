"""
Functions to construct `PlotSpec` for common `lstm_ee` evaluation plots.
"""

from lstm_ee.plot.plot_spec import PlotSpec

def get_plot_spec_fom(energy, bins = 200, range_x = (-1, 1)):
    """Create `PlotSpec` for the energy resolution plot."""
    return PlotSpec(
        title   = None,
        label_x = '(Reco - True) / True %s Energy' % (energy),
        label_y = 'Events',
        bins_x  = bins,
        range_x = range_x,
    )

def get_plot_spec_rel_res_vs_true(
    energy, bins = 150, range_x = (0, 5), range_y = (-1, 1)
):
    """Create `PlotSpec` for the 2d hist of rel. energy resolution vs TrueE."""
    return PlotSpec(
        title   = None,
        label_x = 'True %s Energy [GeV]' % (energy),
        label_y = '(Reco - True) / True %s Energy' % (energy),
        bins_x  = bins,
        range_x = range_x,
        bins_y  = bins,
        range_y = range_y
    )

def get_plot_spec_hist(
    energy, bins = 50, range = (0, 5), ratio_range_y = None
):
    """Create `PlotSpec` for energy distribution plot."""
    # pylint: disable=redefined-builtin
    return PlotSpec(
        title         = None,
        label_x       = '%s Energy [GeV]' % (energy),
        label_y       = 'Events',
        bins_x        = bins,
        range_x       = range,
        bins_y        = None,
        range_y       = None,
        ratio_range_y = ratio_range_y
    )

def get_plot_spec_binstat_abs(energy, bins = 50, range = (0, 5)):
    """
    Create `PlotSpec` for plot of some stat of abs energy resolution vs TrueE.
    """
    # pylint: disable=redefined-builtin
    return PlotSpec(
        title   = None,
        label_x = 'True %s Energy [GeV]' % (energy),
        label_y = '(Reco - True) %s Energy [GeV]' % (energy),
        bins_x  = bins,
        range_x = range,
        bins_y  = None,
        range_y = None,
        minor   = True,
        grid    = True,
    )

def get_plot_spec_binstat_rel(energy, bins = 50, range = (0, 5)):
    """
    Create `PlotSpec` for plot of some stat of rel energy resolution vs TrueE.
    """
    # pylint: disable=redefined-builtin
    return PlotSpec(
        title   = None,
        label_x = 'True %s Energy [GeV]' % (energy),
        label_y = '(Reco - True) / True %s Energy' % (energy),
        bins_x  = bins,
        range_x = range,
        bins_y  = None,
        range_y = None,
        minor   = True,
        grid    = True,
    )


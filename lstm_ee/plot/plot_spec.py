"""
Definition of a class that specifies plot axes/style.
"""

import copy
import numpy as np

class PlotSpec():
    """A class that specifies plot axes and style.

    Parameters
    ----------
    title : str or None, optional
        Plot title.
    label_x : str or None, optional
        Label of a horizontal axis.
    label_y : str or None, optional
        Label of a vertical axis.
    bins_x : int or list of int or None, optional
        Bins specification for the horizontal axis (in case of histogram plot).
    bins_y : int or list of int or None, optional
        Bins specification for the vertical axis (in case of histogram plot).
    range_x : (float, float) or None, optional
        Horizontal range of a plot.
    range_y : (float, float) or None, optional
        Vertical range of a plot.
    grid : bool or None, optional
        If True than the axes grid will be drawn.
    minor : bool or None, optional
        If True then minor axes ticks will be shown. If `grid` is also True
        then the minor grid will be drawn.
    ratio_range_y : (float, float) or None,
        If a plot has associated ratio plot then this parameter will control
        the vertical range of the ratio plot.
    """

    # pylint: disable=no-member
    __slots__ = (
        'title',
        'label_x',
        'label_y',
        'bins_x',
        'bins_y',
        'range_x',
        'range_y',
        'grid',
        'minor',
        'ratio_range_y',
    )

    def __init__(self, **kwargs):

        for k in self.__slots__:
            setattr(self, k, None)

        for k,v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)

        self.init_bins()

        # pylint: disable=access-member-before-definition
        if self.ratio_range_y is None:
            self.ratio_range_y = (0.90, 1.10)

    def copy(self):
        """Make a copy of itself"""
        return copy.deepcopy(self)

    @staticmethod
    def calc_bins(bins, range):
        """Calculate bin edges based on a number of bins and range."""
        # pylint: disable=redefined-builtin
        if bins is None:
            return None

        if isinstance(bins, int) and (range is not None):
            return np.linspace(*range, num = bins + 1, endpoint = True)

        return bins

    def init_bins(self):
        """Initialize self.bins_*"""
        self.bins_x = PlotSpec.calc_bins(self.bins_x, self.range_x)
        self.bins_y = PlotSpec.calc_bins(self.bins_y, self.range_y)

    def _setup_grid_ticks(self, ax):
        """Add grid lines and ticks to the axes `ax`"""

        if self.grid:
            ax.grid(
                True, which = 'major', linestyle = 'dashed', linewidth = 1.0
            )

        if self.minor:
            ax.minorticks_on()

            if self.grid:
                ax.grid(
                    True, which = 'minor', linestyle = 'dashed',
                    linewidth = 0.5
                )

    def decorate(self, ax, ratio_plot = None):
        """Decorate axes `ax` based on a config stored at `self`"""
        if self.title is not None:
            ax.set_title(self.title)

        if (self.label_x is not None) and (ratio_plot is None):
            ax.set_xlabel(self.label_x)

        if self.label_y is not None:
            ax.set_ylabel(self.label_y)

        if self.range_x is not None:
            ax.set_xlim(self.range_x)

        if self.range_y is not None:
            ax.set_ylim(self.range_y)

        self._setup_grid_ticks(ax)

    def decorate_ratio(self, axr, ratio_plot = None):
        """Decorate ratio plot axes `axr` based on a config stored at `self`"""

        if self.label_x is not None:
            axr.set_xlabel(self.label_x)

        if isinstance(ratio_plot, str) and (ratio_plot.lower() == 'fixed'):
            axr.set_ylim(self.ratio_range_y)

        self._setup_grid_ticks(axr)


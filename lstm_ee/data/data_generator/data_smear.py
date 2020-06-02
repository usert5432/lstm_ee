"""
A definition of a decorator that adds normal smearing to input values.
"""

import numpy as np

from .idata_decorator import IDataDecorator
from .data_noise      import calc_var_indices

class DataSmear(IDataDecorator):
    """A decorator around `IDataGenerator` that adds normal smearing to inputs.

    This decorator modifies input variables that decorated object produces by
    adding normal noise to them. Unlike `DataNoise` that adds the same noise to
    all variables in a slice, `DataSmear` samples noise for each variable in a
    slice independently.

    The noise is added multiplicatively to inputs.

    Parameters
    ----------
    dgen : IDataGenerator
        `IDataGenerator` to be decorated.
    smear : float
        Sigma of the normal distribution that will be used to sample noise.
    affected_vars_slice : list of str or None, optional
        List of slice level variables that will be affected by noise.
    affected_vars_png2d : list of str or None, optional
        List of 2D prong level variables that will be affected by noise.
    affected_vars_png3d : list of str or None, optional
        List of 3D prong level variables that will be affected by noise.

    See Also
    --------
    DataNoise
    """

    def __init__(
        self, dgen,
        smear               = None,
        affected_vars_slice = None,
        affected_vars_png2d = None,
        affected_vars_png3d = None,
    ):
        super(DataSmear, self).__init__(dgen)

        self._smear      = smear
        self._vars_slice = affected_vars_slice
        self._vars_png3d = affected_vars_png3d
        self._vars_png2d = affected_vars_png2d

        self._vars_idx_slice = calc_var_indices(
            self.vars_input_slice, affected_vars_slice
        )
        self._vars_idx_png2d = calc_var_indices(
            self.vars_input_png2d, affected_vars_png2d
        )
        self._vars_idx_png3d = calc_var_indices(
            self.vars_input_png3d, affected_vars_png3d
        )

    def _apply_smear(self, input_values, var_idx):

        if input_values.size == 0:
            return

        smear = np.random.normal(
            loc   = 1.0,
            scale = self._smear,
            size  = input_values.shape[:-1] + (len(var_idx),)
        )

        input_values[..., var_idx] *= smear

    def __getitem__(self, index):

        batch_data = self._dgen[index]
        inputs     = batch_data[0]

        if self._vars_slice is not None:
            self._apply_smear(inputs['input_slice'], self._vars_idx_slice)

        if self._vars_png2d is not None:
            self._apply_smear(inputs['input_png2d'], self._vars_idx_png2d)

        if self._vars_png3d is not None:
            self._apply_smear(inputs['input_png3d'], self._vars_idx_png3d)

        return batch_data


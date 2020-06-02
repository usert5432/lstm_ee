"""
A definition of a decorator that adds noise to input values.
"""

from .idata_decorator import IDataDecorator
from .funcs.noise import select_noise

def calc_var_indices(input_vars, affected_vars):
    """Calculate indices of `affected_vars` in `input_vars`"""
    if affected_vars is None:
        return None

    return [ input_vars.index(var) for var in affected_vars ]

class DataNoise(IDataDecorator):
    """A decorator around `IDataGenerator` that adds noise to inputs.

    This decorator modifies input variables that decorated object produces by
    adding random noise to them. Random value is sampled once per slice and
    applied to all variables specified at affected_vars_*.

    The noise is added multiplicatively to inputs.

    Parameters
    ----------
    dgen : IDataGenerator
        `IDataGenerator` to be decorated.
    noise : { None, 'uniform', 'gaussian', 'discrete', 'debug' }
        Type of noise to be added.
    noise_kwargs : dict or None, optional
        Noise parameters. C.f. `select_noise`.
    affected_vars_slice : list of str or None, optional
        List of slice level variables that will be affected by noise.
    affected_vars_png2d : list of str or None, optional
        List of 2D prong level variables that will be affected by noise.
    affected_vars_png3d : list of str or None, optional
        List of 3D prong level variables that will be affected by noise.

    See Also
    --------
    lstm_ee.data.data_generator.funcs.noise.select_noise
    """

    def __init__(
        self, dgen,
        noise               = None,
        noise_kwargs        = None,
        affected_vars_slice = None,
        affected_vars_png2d = None,
        affected_vars_png3d = None,
    ):
        super(DataNoise, self).__init__(dgen)

        if noise_kwargs is None:
            noise_kwargs = {}

        self._noise      = select_noise(noise, **noise_kwargs)
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

    @staticmethod
    def _apply_noise(input_values, var_idx, noise):
        """
        Apply noise to vars specified by `var_idx` in `input_values` inplace
        """

        if input_values.size == 0:
            return

        broadcasted_shape = noise.shape + (1,) * (input_values.ndim - 1)
        broadcasted_noise = noise.reshape(broadcasted_shape)

        input_values[..., var_idx] *= (1 + broadcasted_noise)

    def _get_noise(self, inputs):
        batch_sizes = [ x.shape[0] for x in inputs.values() ]
        batch_size  = batch_sizes[0]

        return self._noise.get(batch_size)

    def __getitem__(self, index):

        batch_data = self._dgen[index]
        inputs     = batch_data[0]
        noise      = self._get_noise(inputs)#.ravel()

        if self._vars_slice is not None:
            DataNoise._apply_noise(
                inputs['input_slice'], self._vars_idx_slice, noise
            )

        if self._vars_png2d is not None:
            DataNoise._apply_noise(
                inputs['input_png2d'], self._vars_idx_png2d, noise
            )

        if self._vars_png3d is not None:
            DataNoise._apply_noise(
                inputs['input_png3d'], self._vars_idx_png3d, noise
            )

        return batch_data



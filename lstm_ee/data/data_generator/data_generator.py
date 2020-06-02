"""
Definition of a DataGenerator that creates data batches from DataLoader
"""

import math
import numpy as np

from .funcs.funcs_varr import unpack_varr_arrays
from .idata_generator  import IDataGenerator

class DataGenerator(IDataGenerator):
    """Primary `lstm_ee` DataGenerator that batches data from a `IDataLoader`.

    This generator takes an instance of `IDataLoader` and specification of
    input/target variables and creates batches of data based on them.
    Batches can be retrieved with __getitem__ method that takes batch
    index as an input.

    Parameters
    ----------
    data_loader : `IDataLoader`
        `IDataLoader` which will be used to retrieve values of variables.
    batch_size : int
        Size of the batches to be generated.
    max_prongs : int or None, optional
        If `max_prongs` is not None, then the number of 2D and 3D prongs will
        be truncated by `max_prongs`. Default: None.
    vars_input_slice : list of str or None, optional
        Names of slice level input variables in `data_loader`.
        If None no slice level inputs will be generated. Default: None.
    vars_input_png3d : list of str or None, optional
        Names of 3d prong level input variables in `data_loader`.
        If None no 3d prong level inputs will be generated. Default: None.
    vars_input_png2d : list of str or None, optional
        Names of 2d prong level input variables in `data_loader`.
        If None no 2d prong level inputs will be generated. Default: None.
    var_target_total : str or None, optional
        Name of the variable in `data_loader` that holds total energy of
        the event (e.g. neutrino energy).
        If None, no total energy target will be generated. Default: None
    var_target_primary : str or None, optional
        Name of the variable in `data_loader` that holds primary energy of
        the event (e.g. lepton energy).
        If None, no primary energy target will be generated. Default: None
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        data_loader,
        batch_size         = 1024,
        max_prongs         = None,
        vars_input_slice   = None,
        vars_input_png3d   = None,
        vars_input_png2d   = None,
        var_target_total   = None,
        var_target_primary = None,
    ):
        super(DataGenerator, self).__init__()

        self._data_loader  = data_loader
        self._batch_size   = batch_size
        self._max_prongs   = max_prongs

        self._vars_input_slice   = vars_input_slice
        self._vars_input_png3d   = vars_input_png3d
        self._vars_input_png2d   = vars_input_png2d
        self._var_target_total   = var_target_total
        self._var_target_primary = var_target_primary

    def get_scalar_data(self, variables, index):
        """Generate batch of scalar data.

        Parameters
        ----------
        variables : list of str
            List of variables names which values will be joined into a batch.
        index : int or list of int or None
            Index that defines slice of values to be used when generating
            batch. If None, all available values will be joined into a batch.

        Returns
        -------
        ndarray, shape (N_SAMPLE, len(variables))
            Values of `variables` with sliced by `index` batched together.
        """
        result = np.empty((len(index), len(variables)), dtype = np.float32)

        for idx,vname in enumerate(variables):
            result[:, idx] = self._data_loader.get(vname, index)

        return result

    def get_varr_data(self, variables, index, max_prongs = None):
        """Generate batch of variable length arrays (prong) data.

        All variables length arrays will be batches together into a fixed
        size `np.ndarray`. Missing variable length values will be padded
        by `np.nan`

        Parameters
        ----------
        variables : list of str
            List of variables names which values will be joined into a batch.
        index : int or list of int or None
            Index that defines slice of values to be used when generating
            batch. If None, all available values will be joined into a batch.
        max_prongs : int or None, optional
            If `max_prongs` is not None, then the variable length dimension
            will be truncated by `max_prongs`.

        Returns
        -------
        ndarray, shape (N_SAMPLE, N_VARR, len(variables))
            Values of `variables` with sliced by `index` and batched together.
            Second dimension goes along the variable length axis.

        See Also
        --------
        unpack_varr_arrays
        """
        result = unpack_varr_arrays(
            self._data_loader, variables, index, max_prongs
        )

        return result

    def get_data(self, index):
        """Generate batch of inputs and targets.

        Only variables from vars_input_* will be used to generate input
        batches. Similarly, only variables from var_target_* will be used
        to generate target batches.

        Parameters
        ----------
        index : int or list of int or None
            Index of the `IDataLoader` this generator holds that specifies
            slice of values to be batched together.
            If None, all available values will be batched.

        Returns
        -------
        (inputs, targets)
            Dictionaries of input and target batches.

        See Also
        --------
        DataGenerator.__getitem__
        """

        inputs  = {}
        targets = {}

        if self._vars_input_slice is not None:
            inputs['input_slice'] = self.get_scalar_data(
                self._vars_input_slice, index
            )

        if self._vars_input_png3d is not None:
            inputs['input_png3d'] = self.get_varr_data(
                self._vars_input_png3d, index, self._max_prongs
            )

        if self._vars_input_png2d is not None:
            inputs['input_png2d'] = self.get_varr_data(
                self._vars_input_png2d, index, self._max_prongs
            )

        if self._var_target_total is not None:
            targets['target_total'] = self.get_scalar_data(
                [ self._var_target_total ], index
            )

        if self._var_target_primary is not None:
            targets['target_primary'] = self.get_scalar_data(
                [ self._var_target_primary ], index
            )

        return (inputs, targets)

    def __len__(self):
        return math.ceil(len(self._data_loader) / self._batch_size)

    @property
    def weights(self):
        return np.ones(len(self._data_loader))

    def __getitem__(self, index):
        start = index * self._batch_size
        end   = min((index + 1) * self._batch_size, len(self._data_loader))

        data_slice    = np.arange(start, end)
        batch_data    = self.get_data(data_slice)
        batch_weights = np.ones(end - start)

        return batch_data + ( [batch_weights, ] * len(batch_data[1]), )


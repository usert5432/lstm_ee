"""
Definition of a DataGenerator interface.
"""

class IDataGenerator:
    """An interface to a DataGenerator.

    An `IDataGenerator` holds an instance of `IDataLoader` object and creates
    batches of data from it.
    """

    def __init__(self):
        self._vars_input_slice   = None
        self._vars_input_png2d   = None
        self._vars_input_png3d   = None
        self._var_target_total   = None
        self._var_target_primary = None
        self._data_loader        = None
        self._weights            = None

    @property
    def vars_input_slice(self):
        """List of slice level input var names to create batches for"""
        return self._vars_input_slice

    @property
    def vars_input_png2d(self):
        """List of 2D prong level input var names to create batches for"""
        return self._vars_input_png2d

    @property
    def vars_input_png3d(self):
        """List of 3D prong level input var names to create batches for"""
        return self._vars_input_png3d

    @property
    def var_target_total(self):
        """Name of total energy target var to create batches for"""
        return self._var_target_total

    @property
    def var_target_primary(self):
        """Name of primary energy target var to create batches for"""
        return self._var_target_primary

    @property
    def data_loader(self):
        """`IDataLoader` values from which will be used to create batches"""
        return self._data_loader

    @property
    def weights(self):
        """`np.ndarray` of sample weights, shape (len(self.data_loader),)"""
        return self._weights

    def __len__(self):
        """Number of batches this `IDataGenerator` is capable of generating"""
        raise NotImplementedError

    def __getitem__(self, index):
        """Get batch with index `index`.

        Returns a batch constructed from `self.data_loader` with index `index`.

        Parameters
        ----------
        index : int
            Batch index. 0 <= `index` < len(self)

        Returns
        -------
        inputs : dict
            Dictionary of batches of input variables where keys are input
            labels: [ 'input_slice', 'input_png3d', 'input_png2d' ] and values
            are the batches themselves.
            If self.vars_input_* is None then the corresponding input will be
            missing from `inputs`.
        targets : dict
            Dictionary of batches of target variables where keys are target
            labels: [ 'target_total', 'target_primary' ] and values are the
            batches themselves.
            If self.var_target_* is None then the corresponding targets will be
            missing from `targets`.
        weight : list of ndarray
            List of weights for each target is `targets`.
        """

        raise NotImplementedError


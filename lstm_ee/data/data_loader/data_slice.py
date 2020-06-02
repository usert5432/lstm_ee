"""
Definition of a `IDataLoader` transformation that takes slice of the values.
"""

import numpy as np
from .idata_loader_decorator import IDataLoaderDecorator

class DataSlice(IDataLoaderDecorator):
    """decorator around `IDataLoader` that keeps only slice of values.

    `DataSlice` creates a "view" of the `IDataLoader` objects that it
    decorates, such that only values with indices specified at the
    `indices` parameter are kept.
    It implements analog of `ndarray`[`indices`] for the `IDataLoader` API.

    Parameters
    ----------
    data_loader : `IDataLoader`
        DataLoader to decorate.
    indices : list of int
        Indices to keep.
    """

    def __init__(self, data_loader, indices):
        super(DataSlice, self).__init__(data_loader)
        self._indices = np.array(indices)

    def __len__(self):
        return len(self._indices)

    def get(self, var, index = None):

        if index is None:
            base_index = self._indices
        else:
            base_index = self._indices[index]

        return self._data_loader.get(var, base_index)


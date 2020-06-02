"""
Definition of a DataLoader shuffling transformation.
"""

import numpy as np

from .idata_loader_decorator import IDataLoaderDecorator

class DataShuffle(IDataLoaderDecorator):
    """decorator around `IDataLoader` that acts like a shuffle transformation.

    `DataShuffle` holds a shuffled list of the `IDataLoader` indices that it
    decorates.  When the user requests some values with `get` function,
    `DataShuffle` transparently maps indices that user requested to the
    shuffled indices. Then, it uses mapped indices to call `get` of the
    `IDataLoader` that it decorates and returns result to the user.

    Parameters
    ----------
    data_loader : `IDataLoader`
        A DataLoader object to be shuffled.
    seed : int
        Seed that is used to initialize PRG.
    """

    def __init__(self, data_loader, seed):
        super(DataShuffle, self).__init__(data_loader)

        self._seed    = seed
        self._indices = np.arange(len(data_loader))

        # TODO: Use separate PRG for this task.
        np.random.seed(seed)
        np.random.shuffle(self._indices)

    def get(self, var, index = None):

        if index is None:
            base_index = self._indices
        else:
            base_index = self._indices[index]

        return self._data_loader.get(var, base_index)


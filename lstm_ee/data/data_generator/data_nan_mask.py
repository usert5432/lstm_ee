"""
A definition of a decorator that replaces NaN values in batches.
"""

import numpy as np

from lstm_ee.consts   import DEF_MASK
from .idata_decorator import IDataDecorator

class DataNANMask(IDataDecorator):
    """A decorator around `IDataGenerator` that fills NaNs in input batches.

    NaNs are replaced by a value of `DEF_MASK`.
    """

    def __getitem__(self, index):
        batch  = self._dgen[index]

        for data in batch[0].values():
            data[np.isnan(data)] = DEF_MASK

        return batch


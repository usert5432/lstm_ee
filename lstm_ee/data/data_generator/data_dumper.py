"""
A definition of a decorator that dumps data batches to disk.
"""

import os
import pickle

from .idata_decorator import IDataDecorator

class DataDumper(IDataDecorator):
    """A decorator around `IDataGenerator` that dumps batches to disk

    This decorator saves batches produces by an object it decorates to a
    disk. Useful for debugging purposes.

    Parameters
    ----------
    dgen : IDataGenerator
        `IDataGenerator` to be decorated.
    outdir : str
        Directory path where batches will be saved.
    """

    def __init__(self, dgen, outdir):
        super(DataDumper, self).__init__(dgen)

        self._outdir = outdir
        os.makedirs(outdir, exist_ok = True)

    def __getitem__(self, index):

        batch_data = self._dgen[index]
        fname      = os.path.join(self._outdir, 'batch_%d.pkl' % (index))

        with open(fname, 'wb') as f:
            pickle.dump(batch_data, f)

        return batch_data


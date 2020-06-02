"""
Definition of the DataLoader for working with HDF files.
"""

import tables
import numpy as np

from .idata_loader import IDataLoader

class HDFLoader(IDataLoader):
    """DataLoader for loading data from the HDF files.

    This is a simple wrapper around hdf5 files exposing `IDataLoader` API
    for accessing data. The hdf5 files are assumed to contain all arrays
    in the root '/' (e.g. '/array1', '/array2', etc). Names of arrays
    will be treated as variable names (variables = [ 'array1', 'array2',... ])
    Each array can either be 1D array of scalars (slice data), or a 1D array of
    variable length arrays (prong data).

    Parameters
    ----------
    path : str
        Path to the hdf file with the dataset.

    Notes
    -----
    HDF5 files have absolutely terrible random access performance. You should
    consider using parallelization when working with HDF5 files.

    Also, quite surprisingly, xz compressed CSV files take much less disk space
    than the compressed HDF files using internal HDF compressors.
    """

    def __init__(self, path):
        # pylint: disable=unused-argument
        super(HDFLoader, self).__init__()

        self._fname = path
        self._f     = tables.open_file(path, 'r')

        nodes = self._f.list_nodes('/')
        self._variables = [ node.name for node in nodes ]

        if not self._variables:
            self._len = 0
        else:
            self._len = len(nodes[0])

    def _lazy_load(self):
        if self._f is None:
            self._f = tables.open_file(self._fname, 'r')

    def __getstate__(self):
        """Serialize object for pickle.

        This function is called when `HDFLoader` is serialized by `pickle`.

        Notes
        -----
        Pickling is required for multiprocessing.

        Internal `tables.File` object cannot be pickled, so we first drop it
        and then reload when needed on the first use.
        """

        if self._f is not None:
            self._f.close()
            self._f = None

        return self.__dict__

    def variables(self):
        return self._variables

    def __len__(self):
        return self._len

    def get(self, var, index = None):
        self._lazy_load()

        node = self._f.get_node('/' + var)

        if index is None:
            return np.array(node[:])
        else:
            return np.array(node[index])


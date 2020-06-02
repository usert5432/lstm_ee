"""Test correctness of hdf files parsing with `HDFLoader`"""

import os
import unittest
import tempfile

import tables
import numpy as np

from lstm_ee.data.data_loader.hdf_loader import HDFLoader
from .tests_data_loader_base import TestsDataLoaderBase

def create_hdf_data_bytes(fname, data):
    """Save HDF5 representation of data from a `data` dict to a file `fname`"""

    def export_values(f, name, values):
        if isinstance(values[0], list):
            node = f.create_vlarray(
                '/', name, tables.Float32Atom(shape=())
            )

            for x in values:
                node.append(np.array(x, dtype = np.float32))
        else:
            f.create_array('/', name, obj = values)

    f = tables.open_file(fname, 'w')

    for c in data.keys():
        export_values(f, c, data[c])

    f.close()

class TestsHDFLoader(TestsDataLoaderBase, unittest.TestCase):
    """Test `HDFLoader` data parsing"""

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        TestsDataLoaderBase.__init__(self)

        self._to_cleanup = []

    def __del__(self):
        for fname in self._to_cleanup:
            os.unlink(fname)

    def _create_data_loader(self, data):

        with tempfile.NamedTemporaryFile('wb', delete = False) as f:
            fname = f.name
            self._to_cleanup.append(fname)

            create_hdf_data_bytes(fname, data)

        return HDFLoader(fname)

if __name__ == '__main__':
    unittest.main()


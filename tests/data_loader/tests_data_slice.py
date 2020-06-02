"""Test `IDataLoader` data slicing by a slicing decorator `DataSlice`"""

import unittest

from lstm_ee.data.data_loader.dict_loader import DictLoader
from lstm_ee.data.data_loader.data_slice  import DataSlice

from .tests_data_loader_base import FuncsDataLoaderBase

class TestsDataSlice(unittest.TestCase, FuncsDataLoaderBase):
    """Test `DataSlice` decorator"""

    def test_scalar_var(self):
        """Test slicing of a single scalar variable"""
        data        = { 'var' : [ 1, 2, 3, 4, -1 ] }
        slice_index = [ 0, 2, 3 ]
        slice_data  = { 'var' : [ 1, 3, 4 ] }
        data_loader = DataSlice(DictLoader(data), slice_index)

        self._compare_scalar_vars(slice_data, data_loader, 'var')

    def test_varr_var(self):
        """Test slicing of a single variable length array variable"""
        data        = { 'var' : [ [1, 2], [], [3], [4,5,6,7], [-1] ] }
        slice_index = [ 0, 4 ]
        slice_data  = { 'var' : [ [1, 2], [-1] ] }
        data_loader = DataSlice(DictLoader(data), slice_index)

        self._compare_varr_vars(slice_data, data_loader, 'var')

if __name__ == '__main__':
    unittest.main()


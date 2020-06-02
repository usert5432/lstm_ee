"""Test `IDataLoader` data shuffling by a `DataShuffle` decorator"""

import unittest
import numpy as np

from lstm_ee.data.data_loader.dict_loader  import DictLoader
from lstm_ee.data.data_loader.data_shuffle import DataShuffle

from .tests_data_loader_base import FuncsDataLoaderBase

class TestsDataShuffle(unittest.TestCase, FuncsDataLoaderBase):
    """Test `DataShuffle` decorator"""

    def test_scalar_var(self):
        """Test shuffling of a single scalar variable"""
        seed        = 1223
        data        = { 'var' : np.array([ 1, 2, 3, 4, -1 ]) }
        data_loader = DataShuffle(DictLoader(data), seed)

        indices_original = np.arange(0, len(data['var']))
        indices_shuffled = np.array(indices_original[:])

        np.random.seed(seed)
        np.random.shuffle(indices_shuffled)
        self.assertTrue(
            np.any(~ np.isclose(indices_original, indices_shuffled))
        )

        data_shuffled = { 'var' : data['var'][indices_shuffled] }
        self._compare_scalar_vars(data_shuffled, data_loader, 'var')

    def test_varr_var(self):
        """Test shuffling of a single variable length array variable"""
        seed        = 321
        data        = { 'var' : np.array([[1, 2], [], [3], [4,5,6,7], [-1]]) }
        data_loader = DataShuffle(DictLoader(data), seed)

        indices_original = np.arange(0, len(data['var']))
        indices_shuffled = np.array(indices_original[:])

        np.random.seed(seed)
        np.random.shuffle(indices_shuffled)
        self.assertTrue(
            np.any(~ np.isclose(indices_original, indices_shuffled))
        )

        data_shuffled = { 'var' : data['var'][indices_shuffled] }
        self._compare_varr_vars(data_shuffled, data_loader, 'var')

if __name__ == '__main__':
    unittest.main()


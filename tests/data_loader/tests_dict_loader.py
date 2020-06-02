"""Test correctness of `DictLoader` constructor"""

import unittest

from lstm_ee.data.data_loader.dict_loader import DictLoader
from .tests_data_loader_base import TestsDataLoaderBase

class TestsDictLoader(TestsDataLoaderBase, unittest.TestCase):
    """Test `DictLoader` correctness"""

    def _create_data_loader(self, data):
        return DictLoader(data)

if __name__ == '__main__':
    unittest.main()


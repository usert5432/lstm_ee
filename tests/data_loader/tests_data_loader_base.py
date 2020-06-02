"""A template for correctness of DataLoader parsing tests"""

import numpy as np

from ..data import (
    TEST_DATA, TEST_TARGET_VAR_TOTAL, TEST_TARGET_VAR_PRIMARY,
    TEST_INPUT_VARS_SLICE, TEST_INPUT_VARS_PNG3D, TEST_INPUT_VARS_PNG2D
)

class FuncsDataLoaderBase():
    """Functions to compare data parsed by IDataLoader with the actual data"""
    # pylint: disable=no-member

    def _retrieve_test_null_data(self, data, data_loader, var, index):
        """Extract (test, null) datasets from the DataLoader and data dict"""
        self.assertEqual(len(data_loader), len(data[var]))

        data_test = data_loader.get(var, index)

        if index is None:
            data_null = np.array(data[var])
        else:
            data_null = np.array(data[var])[index]

        self.assertEqual(len(data_test), len(data_null))

        return (data_test, data_null)

    def _compare_scalar_vars(self, data, data_loader, var, index = None):

        data_test, data_null = self._retrieve_test_null_data(
            data, data_loader, var, index
        )

        self.assertTrue(np.all(np.isclose(data_test, data_null)))

    def _compare_varr_vars(self, data, data_loader, var, index = None):

        data_test, data_null = self._retrieve_test_null_data(
            data, data_loader, var, index
        )

        # pylint: disable=consider-using-enumerate
        for i in range(len(data_test)):
            self.assertTrue(np.all(np.isclose(data_test[i], data_null[i])))

class TestsDataLoaderBase(FuncsDataLoaderBase):
    """A collection of IDataLoader parsing tests

    This template relies on the `self._create_data_loader` that will create
    an `IDataLoader` to be tested from the dictionary of values.
    """
    # pylint: disable=no-member
    # pylint: disable=assignment-from-no-return

    def _create_data_loader(self, data):
        # pylint: disable=no-self-use
        raise RuntimeError("Not Implemented")

    def test_scalar_var(self):
        """Test correctness of parsing of a single scalar variable"""
        data        = { 'var' : [ 1, 2, 3, 4, -1 ] }
        data_loader = self._create_data_loader(data)

        self.assertEqual(data_loader.variables(), ['var'])
        self._compare_scalar_vars(data, data_loader, 'var')

    def test_varr_var(self):
        """
        Test correctness of parsing of a single variable length array variable
        """
        data        = { 'var' : [ [1, 2], [], [3], [4,5,6,7], [-1] ] }
        data_loader = self._create_data_loader(data)

        self.assertEqual(data_loader.variables(), ['var'])
        self._compare_varr_vars(data, data_loader, 'var')

    def test_global_data(self):
        """Test correctness of parsing of the default dataset `TEST_DATA`"""
        data_loader = self._create_data_loader(TEST_DATA)

        self._compare_scalar_vars(
            TEST_DATA, data_loader, TEST_TARGET_VAR_TOTAL
        )

        self._compare_scalar_vars(
            TEST_DATA, data_loader, TEST_TARGET_VAR_PRIMARY
        )

        for var in TEST_INPUT_VARS_SLICE:
            self._compare_scalar_vars(TEST_DATA, data_loader, var)

        for var in TEST_INPUT_VARS_PNG3D:
            self._compare_varr_vars(TEST_DATA, data_loader, var)

        for var in TEST_INPUT_VARS_PNG2D:
            self._compare_varr_vars(TEST_DATA, data_loader, var)

    def test_scalar_var_with_integer_index(self):
        """Test correctness of slicing of scalar values by integer indices"""
        data        = { 'var' : [ 1, 2, 3, 4, -1 ] }
        data_loader = self._create_data_loader(data)

        self.assertEqual(data_loader.variables(), ['var'])
        self._compare_scalar_vars(data, data_loader, 'var', [1])
        self._compare_scalar_vars(data, data_loader, 'var', [0, 1])
        self._compare_scalar_vars(data, data_loader, 'var', [3, 2])
        self._compare_scalar_vars(data, data_loader, 'var', [4, 1, 2])
        self._compare_scalar_vars(data, data_loader, 'var', [4, 1, 2, 0, 3])

    def test_scalar_var_with_boolean_index(self):
        """Test correctness of slicing of scalar values by boolean indices"""
        data        = { 'var' : [ 1, 2, 3, 4, -1 ] }
        data_loader = self._create_data_loader(data)

        mask1 = [ False, False, False, False, False ]
        mask2 = [ True,  True,  True,  True,  True  ]
        mask3 = [ True,  False, False, False, False ]
        mask4 = [ False, False, False, False, True  ]
        mask5 = [ True,  False, False, True,  True  ]
        mask6 = [ True,  False, True,  False, True  ]

        self.assertEqual(data_loader.variables(), ['var'])
        self._compare_scalar_vars(data, data_loader, 'var', mask1)
        self._compare_scalar_vars(data, data_loader, 'var', mask2)
        self._compare_scalar_vars(data, data_loader, 'var', mask3)
        self._compare_scalar_vars(data, data_loader, 'var', mask4)
        self._compare_scalar_vars(data, data_loader, 'var', mask5)
        self._compare_scalar_vars(data, data_loader, 'var', mask6)


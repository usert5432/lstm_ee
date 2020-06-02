"""Test correctness of the weights calculations"""

import unittest

from lstm_ee.data.data_generator.funcs.weights  import (
    calc_flat_whist, flat_weights
)

from lstm_ee.data.data_loader.dict_loader import DictLoader

from ..data import nan_equal

class TestsWeights(unittest.TestCase):
    """Test correctness of the weights calculations"""

    def test_calc_flat_whist_noclip(self):
        """Test flat weight whist calculation without clipping"""
        bins        = 4
        bins_range  = (0, 200)
        bins_null   = [ 0, 50, 100, 150, 200 ]

        values_null   = [ 1, 11, 111, 121, 4 ]
        # bins: [ (0, 50), (50, 100), (100, 150), (150, 200) ]
        # hist_null     = [ 3,    0,     2,    0 ]
        #
        # ( regularized = hist + 1 )
        # hist_null_reg = [ 4,    1,     3,    1 ]
        #
        # ( whist       = 1 / hist_reg )
        # whist_null    = [ 1/4,  1/1,   1/3,  1/1 ]
        #
        # norm = sum(whist) = 1/4 + 1 + 1/3 + 1
        # norm_null     = 31 / 12
        # ( whist_null    = whist_null / norm_null )
        whist_null    = [ 3/31, 12/31, 4/31, 12/31 ]

        (values_test, whist_test, bins_test) = calc_flat_whist(
            DictLoader({ 'weight' : values_null }), var = 'weight',
                bins = bins, range = bins_range, clip = None
        )

        self.assertTrue(nan_equal(values_test, values_null))
        self.assertTrue(nan_equal(whist_test,  whist_null))
        self.assertTrue(nan_equal(bins_test,   bins_null))

    def test_calc_flat_whist(self):
        """Test flat weight whist calculation with clipping"""
        clip        = 2
        bins        = 4
        bins_range  = (0, 200)
        bins_null   = [ 0, 50, 100, 150, 200 ]

        values_null   = [ 1, 11, 111, 121, 4 ]
        # bins: [ (0, 50), (50, 100), (100, 150), (150, 200) ]
        # hist_null     = [ 3,    0,    2,    0 ]
        #
        # ( regularized = hist + 1 )
        # hist_null_reg = [ 4,    1,    3,    1 ]
        #
        # ( whist = 1 / hist_reg )
        # whist_null    = [ 1/4,  1/1,  1/3,  1/1 ]
        #
        # Adding clipping.
        # Max value is min(whist_null) * clip == 1/4 * 2 = 1/2
        # whist_null    = [ 1/4,  1/2,  1/3,  1/2 ]
        #
        # norm = sum(whist) = 1/4 + 1 + 1/3 + 1
        # norm_null     = 19 / 12
        # ( whist_null = whist_null / norm_null )
        whist_null    = [ 3/19, 6/19, 4/19, 6/19 ]

        (values_test, whist_test, bins_test) = calc_flat_whist(
            DictLoader({ 'weight' : values_null }), var = 'weight',
                bins = bins, range = bins_range, clip = clip
        )

        self.assertTrue(nan_equal(values_test, values_null))
        self.assertTrue(nan_equal(whist_test,  whist_null))
        self.assertTrue(nan_equal(bins_test,   bins_null))

    def test_flat_weights_noclip(self):
        """Test flat weight calculation without clipping"""
        bins        = 4
        bins_range  = (0, 200)
        #bins_null   = [ 0, 50, 100, 150, 200 ]

        # bins: [ (0, 50), (50, 100), (100, 150), (150, 200) ]
        values_null = [ 1, 11, 111, 121, 4 ]
        values_bins = [ 0, 0,  2,   2,   0 ]
        whist_null  = [ 3/31, 12/31, 4/31, 12/31 ]

        weights_null = [ whist_null[i] for i in values_bins ]
        weights_null = [
            x * len(values_null) / sum(weights_null) for x in weights_null
        ]

        weights_test = flat_weights(
            DictLoader({ 'weight' : values_null }), var = 'weight',
                bins = bins, range = bins_range, clip = None
        )

        self.assertTrue(nan_equal(weights_test, weights_null))

    def test_flat_weights(self):
        """Test flat weight calculation with clipping"""
        clip        = 2
        bins        = 4
        bins_range  = (0, 200)
        #bins_null   = [ 0, 50, 100, 150, 200 ]

        # bins: [ (0, 50), (50, 100), (100, 150), (150, 200) ]
        values_null = [ 1, 11, 111, 121, 4 ]
        values_bins = [ 0, 0,  2,   2,   0 ]
        whist_null  = [ 1/4, 1/1, 1/3, 1/1 ]

        weights_null = [ whist_null[i] for i in values_bins ]
        weights_null = [
            x * len(values_null) / sum(weights_null) for x in weights_null
        ]

        weights_test = flat_weights(
            DictLoader({ 'weight' : values_null }), var = 'weight',
                bins = bins, range = bins_range, clip = clip
        )

        self.assertTrue(nan_equal(weights_test, weights_null))

if __name__ == '__main__':
    unittest.main()


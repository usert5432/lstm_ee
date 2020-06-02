"""Test correctness of the noise application by the `DataNoise` decorator"""

import unittest
import numpy as np

from lstm_ee.data.data_generator.data_noise import DataNoise

from .tests_data_generator_base import (
    DataGenerator, DictLoader, TestsDataGeneratorBase
)

class TestsNoise(TestsDataGeneratorBase, unittest.TestCase):
    """Test correctness of the `DataNoise` decorator"""

    @staticmethod
    def _create_shift_noise_dgen(
        data_dict, batch_size, noise_magnitude, dg_var_dict, noise_var_dict
    ):
        dgen = DataGenerator(
            DictLoader(data_dict), batch_size = batch_size, **dg_var_dict,
        )

        dgen = DataNoise(
            dgen,
            noise        = 'debug',
            noise_kwargs = { 'values' : noise_magnitude },
            **noise_var_dict
        )

        return dgen

    def test_single_varr_var_noise_batch_size_1(self):
        """
        Test noise on a single variable length array var when batch_size=1
        """

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]

        data = {
            'var' : [[4,1], [1,2,3,4], [4,3,2,1], [2,1], []]
        }

        batch_size = 1
        batch_data = [
            { 'input_png3d' : [[[scale1*4],[scale1*1]]] },
            { 'input_png3d' : [[[scale1*1],[scale1*2],[scale1*3],[scale1*4]]] },
            { 'input_png3d' : [[[scale1*4],[scale1*3],[scale1*2],[scale1*1]]] },
            { 'input_png3d' : [[[scale1*2],[scale1*1]]] },
            { 'input_png3d' : np.empty((1, 0, 1)) },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_png3d'    : [ 'var' ] },
            { 'affected_vars_png3d' : [ 'var' ] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_single_varr_var_noise_batch_size_3(self):
        """
        Test noise on a single variable length array var when batch_size=3
        """

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]
        scale2 = 1 + noise_magnitude[1]
        scale3 = 1 + noise_magnitude[2]

        data = {
            'var' : [[4,1], [1,2,3,4], [4,3,2,1], [2,1], []]
        }

        batch_size = 3
        batch_data = [
            {
                'input_png3d' : [
                    [[scale1*4], [scale1*1], [np.nan],   [np.nan]  ],
                    [[scale2*1], [scale2*2], [scale2*3], [scale2*4]],
                    [[scale3*4], [scale3*3], [scale3*2], [scale3*1]],
                ]
            },
            {
                'input_png3d' : [
                    [[scale1*2], [scale1*1]],
                    [[np.nan],   [np.nan]  ]
                ]
            },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_png3d'    : [ 'var' ] },
            { 'affected_vars_png3d' : [ 'var' ] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_varr_var_noise_batch_size_1(self):
        """
        Test noise on multiple variable length array vars when batch_size=1
        """

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]

        data = {
            'var1' : [[4,1], [1,2,3,4], [], [2,1], [1]],
            'var2' : [[0,3], [4,3,2,1], [], [2,1], [3]],
            'var3' : [[3,2], [0,9,8,7], [], [1,3], [6]],
        }

        batch_size = 1
        batch_data = [
            {
                'input_png3d' : [
                    [
                        [scale1*4, scale1*0, 3],
                        [scale1*1, scale1*3, 2],
                    ],
                ]
            },
            {
                'input_png3d' : [
                    [
                        [scale1*1, scale1*4, 0],
                        [scale1*2, scale1*3, 9],
                        [scale1*3, scale1*2, 8],
                        [scale1*4, scale1*1, 7],
                    ],
                ]
            },
            { 'input_png3d' : np.empty((1, 0, 3)) },
            {
                'input_png3d' : [
                    [
                        [scale1*2, scale1*2, 1],
                        [scale1*1, scale1*1, 3],
                    ],
                ]
            },
            { 'input_png3d' : [ [ [scale1*1, scale1*3, 6], ], ] },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_png3d'    : [ 'var1', 'var2', 'var3' ] },
            { 'affected_vars_png3d' : [ 'var1', 'var2' ] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_varr_var_noise_batch_size_3(self):
        """
        Test noise on multiple variable length array vars when batch_size=3
        """

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]
        scale2 = 1 + noise_magnitude[1]

        data = {
            'var1' : [[4,1], [1,2,3,4], [], [2,1], [1]],
            'var2' : [[0,3], [4,3,2,1], [], [2,1], [3]],
            'var3' : [[3,2], [0,9,8,7], [], [1,3], [6]],
        }

        batch_size = 3
        batch_data = [
            {
                'input_png3d' : [
                    [
                        [scale1*4, scale1*0, 3],
                        [scale1*1, scale1*3, 2],
                        [np.nan,   np.nan,   np.nan],
                        [np.nan,   np.nan,   np.nan],
                    ],
                    [
                        [scale2*1, scale2*4, 0],
                        [scale2*2, scale2*3, 9],
                        [scale2*3, scale2*2, 8],
                        [scale2*4, scale2*1, 7],
                    ],
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ],
                ]
            },
            {
                'input_png3d' : [
                    [
                        [scale1*2, scale1*2, 1],
                        [scale1*1, scale1*1, 3],
                    ],
                    [
                        [scale2*1, scale2*3, 6],
                        [np.nan,   np.nan,   np.nan],
                    ],
                ]
            },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_png3d'    : [ 'var1', 'var2', 'var3' ] },
            { 'affected_vars_png3d' : [ 'var1', 'var2' ] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_single_scalar_var_noise_batch_size_1(self):
        """Test noise on a single scalar var when batch_size=1"""

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]

        data = { 'var' : [1,2,3,4,5] }

        batch_size = 1
        batch_data = [
            { 'input_slice' : [[scale1*1]] },
            { 'input_slice' : [[scale1*2]] },
            { 'input_slice' : [[scale1*3]] },
            { 'input_slice' : [[scale1*4]] },
            { 'input_slice' : [[scale1*5]] },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_slice'    : [ 'var' ] },
            { 'affected_vars_slice' : [ 'var' ] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_single_scalar_var_noise_batch_size_3(self):
        """Test noise on a single scalar var when batch_size=3"""

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]
        scale2 = 1 + noise_magnitude[1]
        scale3 = 1 + noise_magnitude[2]

        data = { 'var' : [1,2,3,4,5] }

        batch_size = 3
        batch_data = [
            { 'input_slice' : [ [scale1*1], [scale2*2], [scale3*3] ] },
            { 'input_slice' : [ [scale1*4], [scale2*5]] },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_slice'    : [ 'var' ] },
            { 'affected_vars_slice' : [ 'var' ] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_scalar_var_noise_batch_size_1(self):
        """Test noise on multiple scalar vars when batch_size=1"""

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]

        data = {
            'var1' : [1,2,3,4,5],
            'var2' : [6,7,8,9,0],
            'var3' : [3,4,5,6,7],
            'var4' : [8,9,0,1,2],
            'var5' : [5,6,7,6,5],
        }

        batch_size = 1
        batch_data = [
            { 'input_slice' : [[scale1*1, 6, scale1*3, 8, 5]], },
            { 'input_slice' : [[scale1*2, 7, scale1*4, 9, 6]], },
            { 'input_slice' : [[scale1*3, 8, scale1*5, 0, 7]], },
            { 'input_slice' : [[scale1*4, 9, scale1*6, 1, 6]], },
            { 'input_slice' : [[scale1*5, 0, scale1*7, 2, 5]], },
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_slice'    : ['var1','var2','var3','var4','var5'] },
            { 'affected_vars_slice' : ['var1','var3'] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_scalar_var_noise_batch_size_3(self):
        """Test noise on multiple scalar vars when batch_size=3"""

        noise_magnitude = [ 0.10, -0.20, 0.30, -0.10, -0.30 ]
        scale1 = 1 + noise_magnitude[0]
        scale2 = 1 + noise_magnitude[1]
        scale3 = 1 + noise_magnitude[2]

        data = {
            'var1' : [1,2,3,4,5],
            'var2' : [6,7,8,9,0],
            'var3' : [3,4,5,6,7],
            'var4' : [8,9,0,1,2],
            'var5' : [5,6,7,6,5],
        }

        batch_size = 3
        batch_data = [
            {
                'input_slice' : [
                    [scale1*1, 6, scale1*3, 8, 5],
                    [scale2*2, 7, scale2*4, 9, 6],
                    [scale3*3, 8, scale3*5, 0, 7],
                ],
            },
            {
                'input_slice' : [
                    [scale1*4, 9, scale1*6, 1, 6],
                    [scale2*5, 0, scale2*7, 2, 5],
                ]
            }
        ]

        dgen = TestsNoise._create_shift_noise_dgen(
            data, batch_size, noise_magnitude,
            { 'vars_input_slice'    : ['var1','var2','var3','var4','var5'] },
            { 'affected_vars_slice' : ['var1','var3'] },
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

if __name__ == '__main__':
    unittest.main()


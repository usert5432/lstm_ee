"""Test correctness of the prong sorting by the `DataProngSorter` decorator"""

import unittest
import numpy as np

from lstm_ee.data.data_generator import DataProngSorter

from .tests_data_generator_base import (
    DictLoader, DataGenerator, TestsDataGeneratorBase
)

class TestsVarrSorting(TestsDataGeneratorBase, unittest.TestCase):
    """Test correctness of the `DataProngSorter` decorator"""

    @staticmethod
    def _make_sorted_dgen(
        data, batch_size, prong_sorter, input_vars, is3d = True
    ):
        kwargs = {}

        if is3d:
            kwargs = { 'vars_input_png3d' : input_vars }
            input_name = 'input_png3d'
        else:
            kwargs = { 'vars_input_png2d' : input_vars }
            input_name = 'input_png2d'

        dgen = DataGenerator(
            DictLoader(data), batch_size = batch_size, **kwargs
        )

        return DataProngSorter(dgen, prong_sorter, input_name, input_vars)

    def test_single_var_sorter_batch_size_1_ascending(self):
        """
        Test single var prong sorting in ascending order when batch_size=1
        """
        data = {
            'var' : [[4,1], [1,2,3,4], [4,3,2,1], [2,1], []]
        }

        batch_size = 1
        batch_data = [
            { 'input_png3d' : [[[1],[4]]] },
            { 'input_png3d' : [[[1],[2],[3],[4]]] },
            { 'input_png3d' : [[[1],[2],[3],[4]]] },
            { 'input_png3d' : [[[1],[2]]] },
            { 'input_png3d' : np.empty((1, 0, 1)) },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '+var', [ 'var' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_single_var_sorter_batch_size_1_descending(self):
        """
        Test single var prong sorting in descending order when batch_size=1
        """
        data = {
            'var' : [[4,1], [1,2,3,4], [4,3,2,1], [2,1], []]
        }

        batch_size = 1
        batch_data = [
            { 'input_png3d' : [[[4],[1]]] },
            { 'input_png3d' : [[[4],[3],[2],[1]]] },
            { 'input_png3d' : [[[4],[3],[2],[1]]] },
            { 'input_png3d' : [[[2],[1]]] },
            { 'input_png3d' : np.empty((1, 0, 1)) },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '-var', [ 'var' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_single_var_sorter_batch_size_3_ascending(self):
        """
        Test single var prong sorting in ascending order when batch_size=3
        """
        data = {
            'var' : [[4,1], [1,2,3,4], [4,3,2,1], [2,1], []]
        }

        batch_size = 3
        batch_data = [
            {
                'input_png3d' : [
                    [ [1], [4], [np.nan], [np.nan]],
                    [ [1], [2], [3],        [4]],
                    [ [1], [2], [3],        [4]],
                ]
            },
            {
                'input_png3d' : [
                    [ [1],        [2] ],
                    [ [np.nan], [np.nan] ],
                ]
            },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '+var', [ 'var' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_single_var_sorter_batch_size_3_descending(self):
        """
        Test single var prong sorting in descending order when batch_size=3
        """
        data = {
            'var' : [[4,1], [1,2,3,4], [4,3,2,1], [2,1], []]
        }

        batch_size = 3
        batch_data = [
            {
                'input_png3d' : [
                    [ [4], [1], [np.nan], [np.nan]],
                    [ [4], [3], [2],        [1]],
                    [ [4], [3], [2],        [1]],
                ]
            },
            {
                'input_png3d' : [
                    [ [2],        [1] ],
                    [ [np.nan], [np.nan] ],
                ]
            },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '-var', [ 'var' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_var_sorter_batch_size_1_ascending(self):
        """
        Test multi var prong sorting in ascending order when batch_size=1
        """

        data = {
            'data1'  : [[6,3,6,2],[2,5],[1],[4,2],[3,8,4]],
            'sortby' : [[8,9,3,7],[5,1],[3],[2,7],[7,1,3]],
            'data2'  : [[1,6,2,1],[4,2],[2],[8,4],[1,9,8]],
        }

        batch_size = 1
        batch_data = [
            { 'input_png3d' : [ [ [6,3,2], [2,7,1], [6,8,1], [3,9,6] ] ] },
            { 'input_png3d' : [ [ [5,1,2], [2,5,4] ] ] },
            { 'input_png3d' : [ [ [1,3,2] ] ] },
            { 'input_png3d' : [ [ [4,2,8], [2,7,4] ] ] },
            { 'input_png3d' : [ [ [8,1,9], [4,3,8], [3,7,1] ] ] },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '+sortby', [ 'data1', 'sortby', 'data2' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_var_sorter_batch_size_1_descending(self):
        """
        Test multi var prong sorting in descending order when batch_size=1
        """
        data = {
            'data1'  : [[6,3,6,2],[2,5],[1],[4,2],[3,8,4]],
            'sortby' : [[8,9,3,7],[5,1],[3],[2,7],[7,1,3]],
            'data2'  : [[1,6,2,1],[4,2],[2],[8,4],[1,9,8]],
        }

        batch_size = 1
        batch_data = [
            { 'input_png3d' : [ [ [3,9,6], [6,8,1], [2,7,1], [6,3,2] ] ] },
            { 'input_png3d' : [ [ [2,5,4], [5,1,2] ] ] },
            { 'input_png3d' : [ [ [1,3,2] ] ] },
            { 'input_png3d' : [ [ [2,7,4], [4,2,8] ] ] },
            { 'input_png3d' : [ [ [3,7,1], [4,3,8], [8,1,9] ] ] },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '-sortby', [ 'data1', 'sortby', 'data2' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_var_sorter_batch_size_3_ascending(self):
        """
        Test multi var prong sorting in ascending order when batch_size=3
        """
        data = {
            'data1'  : [[6,3,6,2],[2,5],[1],[4,2],[3,8,4]],
            'sortby' : [[8,9,3,7],[5,1],[3],[2,7],[7,1,3]],
            'data2'  : [[1,6,2,1],[4,2],[2],[8,4],[1,9,8]],
        }

        missing = [ np.nan, np.nan, np.nan ]

        batch_size = 3
        batch_data = [
            {
                'input_png3d' : [
                    [ [6,3,2], [2,7,1], [6,8,1], [3,9,6] ],
                    [ [5,1,2], [2,5,4], missing, missing ],
                    [ [1,3,2], missing, missing, missing ],
                ]
            },
            {
                'input_png3d' : [
                    [ [4,2,8], [2,7,4], missing ],
                    [ [8,1,9], [4,3,8], [3,7,1] ],
                ]
            },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '+sortby', [ 'data1', 'sortby', 'data2' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_var_sorter_batch_size_3_descending(self):
        """
        Test multi var prong sorting in descending order when batch_size=3
        """
        data = {
            'data1'  : [[6,3,6,2],[2,5],[1],[4,2],[3,8,4]],
            'sortby' : [[8,9,3,7],[5,1],[3],[2,7],[7,1,3]],
            'data2'  : [[1,6,2,1],[4,2],[2],[8,4],[1,9,8]],
        }

        missing = [ np.nan, np.nan, np.nan ]

        batch_size = 3
        batch_data = [
            {
                'input_png3d' : [
                    [ [3,9,6], [6,8,1], [2,7,1], [6,3,2] ],
                    [ [2,5,4], [5,1,2], missing, missing ],
                    [ [1,3,2], missing, missing, missing ],
                ]
            },
            {
                'input_png3d' : [
                    [ [2,7,4], [4,2,8], missing ],
                    [ [3,7,1], [4,3,8], [8,1,9] ],
                ]
            },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '-sortby', [ 'data1', 'sortby', 'data2' ]
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_multi_var_sorter_batch_size_3_descending_png2d(self):
        """
        Test multi var 2D prong sorting in descending order when batch_size=3
        """
        data = {
            'data1'  : [[6,3,6,2],[2,5],[1],[4,2],[3,8,4]],
            'sortby' : [[8,9,3,7],[5,1],[3],[2,7],[7,1,3]],
            'data2'  : [[1,6,2,1],[4,2],[2],[8,4],[1,9,8]],
        }

        missing = [ np.nan, np.nan, np.nan ]

        batch_size = 3
        batch_data = [
            {
                'input_png2d' : [
                    [ [3,9,6], [6,8,1], [2,7,1], [6,3,2] ],
                    [ [2,5,4], [5,1,2], missing, missing ],
                    [ [1,3,2], missing, missing, missing ],
                ]
            },
            {
                'input_png2d' : [
                    [ [2,7,4], [4,2,8], missing ],
                    [ [3,7,1], [4,3,8], [8,1,9] ],
                ]
            },
        ]

        dgen = TestsVarrSorting._make_sorted_dgen(
            data, batch_size, '-sortby', [ 'data1', 'sortby', 'data2' ],
            False
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)


if __name__ == '__main__':
    unittest.main()


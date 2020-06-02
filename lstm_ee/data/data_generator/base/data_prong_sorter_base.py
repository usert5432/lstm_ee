"""
A definition of a decorator that changes prong order in batches.
"""

import logging
from lstm_ee.data.data_generator.funcs.prong_sorter import (
    SingleVarProngSorter, RandomizedProngSorter
)

LOGGER = logging.getLogger(
    'lstm_ee.data.data_generator.base.data_prong_sorter_base'
)

class DataProngSorterBase:
    """A decorator around DataGenerator that changes order of prongs.

    Parameters
    ----------
    dgen : DataGenerator
        DataGenerator which produces data batches where prongs should be
        reordered.
    name : { "random", "+var_name", "-var_name", None }
        Type of the prong reordering. If None then prong order will not be
        altered. If "+var_name" then prongs will be sorted in ascending order
        by variable "var_name". If "-var_name" then prongs will be sorted
        in descending order by "var_name". If "random" then the prong order
        will be randomized.
    input_name : str
        Name of the input that `dgen` generates which prong order should
        be altered.
    input_vars : list of str
        List of variable names which values `dgen` generates in the input
        called `input_name`.
    """

    def __init__(self, dgen, name, input_name, input_vars):
        self._dgen         = dgen
        self._name         = name
        self._input_name   = input_name
        self._input_vars   = input_vars
        self._prong_sorter = None

        self._init_prong_sorter()

    def _init_prong_sorter(self):
        if self._name is None:
            return

        if not isinstance(self._name, str):
            self._prong_sorter = self._name

        elif self._name == 'random':
            LOGGER.info(
                "Using randomized prong order for '%s'", self._input_name
            )
            self._prong_sorter = RandomizedProngSorter()

        else:
            LOGGER.info(
                "Sorting prongs by '%s' order for '%s'",
                self._name, self._input_name
            )

            self._prong_sorter = SingleVarProngSorter(
                self._name, self._input_vars
            )

    def __getitem__(self, index):
        batch = self._dgen[index]

        if self._prong_sorter is None:
            return batch

        self._prong_sorter(batch[0][self._input_name])

        return batch


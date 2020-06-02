"""
A definition of a decorator that changes prong order in batches.

C.f. `lstm_ee.data.data_generator.base.data_prong_sorter_base`.
"""

from .idata_decorator             import IDataDecorator
from .base.data_prong_sorter_base import DataProngSorterBase

class DataProngSorter(DataProngSorterBase, IDataDecorator):
    # pylint: disable=C0115

    def __init__(self, dgen, name, input_name, input_vars):
        IDataDecorator     .__init__(self, dgen)
        DataProngSorterBase.__init__(self, dgen, name, input_name, input_vars)


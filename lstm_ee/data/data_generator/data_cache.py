"""
Definition of a decorator that caches data batches in RAM.

C.f. `lstm_ee.data.data_generator.base.data_cache_base`.
"""

from .idata_decorator      import IDataDecorator
from .base.data_cache_base import DataCacheBase

class DataCache(DataCacheBase, IDataDecorator):
    # pylint: disable=C0115

    def __init__(self, dgen):
        IDataDecorator.__init__(self, dgen)
        DataCacheBase .__init__(self, dgen)


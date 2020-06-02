"""
Definition of a decorator that caches data batches on disk.

C.f. `lstm_ee.data.data_generator.base.data_disk_cache_base`.
"""

from .idata_decorator           import IDataDecorator
from .base.data_disk_cache_base import DataDiskCacheBase

class DataDiskCache(DataDiskCacheBase, IDataDecorator):
    # pylint: disable=C0115

    def __init__(self, dgen, datadir, **kwargs):
        IDataDecorator   .__init__(self, dgen)
        DataDiskCacheBase.__init__(self, dgen, datadir, **kwargs)


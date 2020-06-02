"""
A definition of a decorator that caches data batches in RAM.
"""

import copy
import logging
import threading

LOGGER = logging.getLogger('lstm_ee.data.data_generator.base.data_cache_base')

class DataCacheBase:
    """A decorator around DataGenerator that caches results in RAM.

    Since joining data in batches is very computationally expensive process
    this decorator caches results in RAM, which allows to significantly improve
    performance when the same data batch is reused multiple times (like during
    the training phase). You should have a good amount of RAM though.

    Parameters
    ----------
    dgen : DataGenerator
        DataGenerator that creates batches to be cached.
    """

    def __init__(self, dgen):
        self._dgen    = dgen
        self._cache   = [ None for i in range(len(dgen)) ]
        self._ncached = 0

        self._lock = threading.Lock()

    def _fetch(self, index):
        LOGGER.debug("Adding batch '%d' into cache", index)
        result = self._dgen[index]

        with self._lock:
            if self._cache[index] is None:
                self._cache[index]  = result
                self._ncached      += 1

        return result

    def __getitem__(self, index):
        with self._lock:
            v = self._cache[index]

        if v is not None:
            return copy.deepcopy(v)

        return copy.deepcopy(self._fetch(index))


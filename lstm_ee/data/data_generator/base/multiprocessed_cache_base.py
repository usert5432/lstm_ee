"""
A definition of a decorator that precomputes batches in concurrent processes.
"""

import copy
import logging

from multiprocessing import Pool

LOGGER = logging.getLogger(
    'lstm_ee.data.data_generator.base.multiprocessed_cache_base'
)

class MultiprocessedCacheBase:
    """A decorator around DataGenerator to precompute batches in parallel.

    This decorator around DataGenerator will spawn multiple concurrent
    processes to generate batches from the decorated object on the first use.
    The precomputed batches will be stored in the RAM cache.

    This may eat all your RAM since each parallel process will get a copy of
    the data.

    Parameters
    ----------
    dgen : DataGenerator
        DataGenerator that creates batches to be precomputed and cached.
    workers : int
        Number of parallel processes to use.
    """

    def __init__(self, dgen, workers = None):
        self._dgen    = dgen
        self._cache   = None
        self._workers = workers

    def __call__(self, index):
        LOGGER.debug("Fetching batch: %d", index)
        return self._dgen[index]

    def __getitem__(self, index):
        if self._cache is None:
            with Pool(processes = self._workers) as pool:
                self._cache = pool.map(self, range(len(self)))

        return copy.deepcopy(self._cache[index])


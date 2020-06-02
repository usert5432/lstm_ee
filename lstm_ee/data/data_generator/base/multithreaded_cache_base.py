"""
A definition of a decorator that precomputes batches in concurrent threads.
"""

import copy
import queue
import logging
import threading

LOGGER = logging.getLogger(
    'lstm_ee.data.data_generator.base.multithreaded_cache_base'
)

class CacheWorker(threading.Thread):
    """Thread worker that precomputes data batches"""

    def __init__(self, cache):
        super(CacheWorker, self).__init__()
        self._cache = cache

    def run(self):
        while True:
            index = self._cache.queue.get()
            if index is None:
                break

            result = self._cache.base_dgen[index]

            self._cache.add_to_cache(index, result)
            self._cache.queue.task_done()

class MultithreadedCacheBase:
    """A decorator around DataGenerator to precompute batches in threads.

    This decorator around DataGenerator will spawn multiple concurrent threads
    to generate batches from the decorated object on the first use. The
    precomputed batches will be stored in the RAM cache. This form of
    precomputation is largely ineffective due to a python GIL.

    Parameters
    ----------
    dgen : DataGenerator
        DataGenerator that creates batches to be precomputed and cached.
    workers : int
        Number of parallel threads to use.
    """

    def __init__(self, dgen, workers):
        self._dgen   = dgen
        self._cached = False
        self._cache  = [ None for i in range(len(dgen)) ]
        self._cache_lock = threading.Lock()

        self._queue   = queue.Queue(len(dgen))
        self._threads = []

        for i in range(workers):
            thread = CacheWorker(self)
            thread.start()
            self._threads.append(thread)

        for i in range(len(dgen)):
            self._queue.put(i)

    @property
    def queue(self):
        """Pending job queue"""
        return self._queue

    @property
    def cache(self):
        """List of cached batches"""
        return self._cache

    @property
    def base_dgen(self):
        """Decorated DataGenerator"""
        return self._dgen

    def add_to_cache(self, index, data):
        """Add precomputed batch `data` to cache at `index`"""
        with self._cache_lock:
            LOGGER.debug("Adding batch '%d' into cache", index)
            self._cache[index] = data

    def __getitem__(self, index):
        if not self._cached:
            self._queue.join()

            for _ in range(len(self._threads)):
                self._queue.put(None)

            for thread in self._threads:
                thread.join()

            self._cached = True

        return copy.deepcopy(self._cache[index])


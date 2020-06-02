"""
A definition of a decorator that caches data batches on a disk.
"""

import os
import copy
import fcntl
import hashlib
import json
import logging
import pickle
import tempfile

LOGGER = logging.getLogger(
    'lstm_ee.data.data_generator.base.data_disk_cache_base'
)

class DataDiskCacheBase:
    """A decorator around DataGenerator that caches results on a disk.

    It turns out that loading data batches from the hard drive is faster
    than building them on the fly from the raw dataset. This decorator
    caches data batches constructed by the DataGenerator on a disk.

    Parameters
    ----------
    dgen : DataGenerator
        DataGenerator that creates batches to be cached.
    datadir : str
        Directory under which cache will be saved. The cache is saved in a
        subdir ".cache".
    **kwargs : dict
        Dictionary of values that uniquely specify the DataGenerator `dgen`
        and will allow to differentiate given disk cache, from disk caches
        for other DataGenerators.
        For example, if one DataGenerator has batch size of 32 and other has
        batch size of 64, then you should pass something like
        { 'batch_size' : 32 } and { 'batch_size' : 64 } in `kwargs` for
        these DataGenerators respectively. Otherwise, cache for the 32 batch
        size DataGenerator might be erroneously reused for the 64 batch
        size DataGenerator.

    Notes
    -----
    Caches on the disk should be cleaned manually. They are stored under
    `datadir`/.cache
    """

    def __init__(self, dgen, datadir, **kwargs):
        self._dgen    = dgen
        self._config  = copy.deepcopy(kwargs)
        self._datadir = datadir

        self._init_cache_dir()

    def _save_cache_config(self):
        """Save cache configuration to a disk if it is missing.

        If the cache configuration is missing in a cache directory this
        function will simply save it.

        Otherwise it will read existing configuration from the disk and compare
        it to the current. There is tiny probability that a given cache
        directory might hold a cache from different configuration. In such
        case this function will throw an exception.

        Raises
        ------
        RuntimeError
            Cache directory that this decorator intends to use contains
            cache from another decorator that is incompatible with current.
            This is caused by the fact that cache directory name is constructed
            from a hash of the cache configuration. Evidently, hash collision
            occurs for two different cache configurations.
        """
        cache_conf_path = os.path.join(self._cache_root, 'config.json')
        lockfile        = cache_conf_path + '.lock'

        config_str = json.dumps(self._config, sort_keys = True, indent = 4)

        with open(lockfile, 'w') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            if os.path.exists(cache_conf_path):
                with open(cache_conf_path, 'rt') as f:
                    old_config_str = f.read()

                    if old_config_str != config_str:
                        raise RuntimeError(
                            "Finally hash collision for cache dirs found"
                            ". Bailing out"
                        )
            else:
                with open(cache_conf_path, 'wt') as f:
                    f.write(config_str)

    def _init_cache_dir(self):
        """Initialize cache directory it does not exist."""
        cachedir = bytes(json.dumps(self._config, sort_keys = True), 'utf-8')
        cachedir = hashlib.sha1(cachedir).hexdigest()

        self._cache_root = os.path.join(
            self._datadir, '.cache', '%s' % (cachedir)
        )
        os.makedirs(self._cache_root, exist_ok = True)

        self._save_cache_config()

    def _get_batch_fname(self, index):
        return os.path.join(self._cache_root, 'batch_%d.pkl' % (index,))

    def _load_batch(self, index):
        batch_fname = self._get_batch_fname(index)

        try:
            with open(batch_fname, 'rb') as f:
                batch = pickle.load(f)
        except IOError:
            return None

        return batch

    def _fetch_batch(self, index):
        """Retrieves batch from the decorated object and saves it to disk."""
        batch       = self._dgen[index]
        batch_fname = self._get_batch_fname(index)

        # To ensure atomicity of writes
        with tempfile.NamedTemporaryFile(
            'wb', dir = self._cache_root, delete = False
        ) as f:
            temp_fname = f.name
            pickle.dump(batch, f)

        os.rename(temp_fname, batch_fname)

        return batch

    def __getitem__(self, index):
        batch = self._load_batch(index)

        if batch is None:
            LOGGER.debug("Cache miss. Fetching batch: %d", index)
            batch = self._fetch_batch(index)

        return batch


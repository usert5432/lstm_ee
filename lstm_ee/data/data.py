"""
A collection of routines to simplify data handling.
"""

import json
import logging
import os

import numpy as np

from lstm_ee.data.data_loader import (
    CSVLoader, HDFLoader, DictLoader, DataShuffle, DataSlice
)
from lstm_ee.data.data_generator import (
    DataCache, DataDiskCache, DataGenerator, DataNANMask, DataNoise,
    DataProngSorter, DataWeight, MultiprocessedCache, MultithreadedCache
)
from lstm_ee.data.data_generator.funcs.weights      import flat_weights

H5_EXTS = [ 'h5', 'hdf', 'hdf5' ]
LOGGER  = logging.getLogger('lstm_ee.data')

def guess_data_loader(path):
    """Find appropriate DataLoader based on a file path

    This function tries to guess proper instance of `IDataLoader` based on a
    file extension.
    """
    if isinstance(path, dict):
        return DictLoader(path)

    if isinstance(path, str):
        for ext in H5_EXTS:
            if path.endswith(ext):
                return HDFLoader(path)

        return CSVLoader(path)

    raise RuntimeError("Unknown how to load data: %s" % (path))

def train_test_split(data_loader, test_size):
    """Split DataLoader into train and test DataLoaders

    Split is done such that first N samples go to the training sample and
    the remaining one go to the test sample. So, shuffle your dataset first
    to avoid possibility of covariate shift between train and test samples.

    Parameters
    ----------
    data_loader : IDataLoader
        An `IDataLoader` that will be split.
    test_size : int or float or None
        Amount of samples from `data_loader` that will land in the test sample.
        If `test_size` is int then `test_size` entries from the `data_loader`
        will go to the test sample.
        If `test_size` is float and `test_size` < 1 then a fraction of
        `test_size` of the `data_loader` will go to the validation sample.
        If None the no split will be performed.

    Returns
    -------
    [ IDataLoader, IDataLoader ]
        A list of train and test DataLoaders.
    """

    if test_size is None:
        return [ data_loader, ]

    indices = np.arange(len(data_loader))

    if test_size <= 1:
        n_train = int(len(data_loader) * (1 - test_size))
    else:
        n_train = max(0, len(data_loader) - int(test_size))

    return [
        DataSlice(data_loader, indices[:n_train]),
        DataSlice(data_loader, indices[n_train:]),
    ]

def construct_data_loader(path, seed, test_size):
    """Load dataset to DataLoader, shuffle it and split into train/test parts.

    Parameters
    ----------
    path : str
        File path from which dataset will be loaded.
    seed : int or None
        Seed that will be used to shuffle data.
    test_size : int or float or None
        Fraction of the dataset that will go to the test sample.
        C.f. `train_test_split` for the detailed description.

    Returns
    -------
    [ IDataLoader, IDataLoader ]
        A list of train and test DataLoaders.

    See Also
    --------
    train_test_split
    """

    data_loader = guess_data_loader(path)
    data_loader = DataShuffle(data_loader, seed)

    return train_test_split(data_loader, test_size)

def add_noise(dgen_list, noise):
    """Add noise decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    noise : list of dict or dict or None
        Noise configuration.
        If dict then `noise` will be simply passed to the `DataNoise`
        constructor.
        If list of dict then multiple `DataNoise` decorators will be applied
        to the `dgen_list`, one for each dict in `noise`.
        If None than `dgen_list` will not be modified.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by `DataNoise`.

    See Also
    --------
    DataNoise
    """

    if noise is None:
        return dgen_list

    if isinstance(noise, list):
        for n in noise:
            dgen_list = add_noise(dgen_list, n)

        return dgen_list

    LOGGER.info(
        "Adding noise: %s",
        json.dumps(noise, sort_keys = True, indent = 4)
    )

    return [ DataNoise(x, **noise) for x in dgen_list ]

def get_weights(weights):
    """Get weight based on configuration in `weights`"""

    if weights is None:
        return None

    if isinstance(weights, str):
        return weights

    name   = weights['name']
    kwargs = weights.get('kwargs', {})

    if name == 'flat':
        # pylint: disable = unnecessary-lambda
        return lambda data_loader : flat_weights(data_loader, **kwargs)

    return weights

def add_cache_decorators(dgen_list, cache, concurrency, workers):
    """Add cache decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    cache : bool or None
        If True then the DataGenerators from `dgen_list` will be cached.
        Otherwise, this function will return unmodified `dgen_list`.
    concurrency : { 'process', 'thread', None }
        Specifies Whether to precompute cache. If None, then cache will not be
        precomputed. Otherwise, it will be precomputed by parallelizing data
        generation in multiple threads/processes.
    workers : int or None
        Number of parallel threads/processes to use for precomputing cache.
        Has no effect if `concurrency` is None.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by cache decorators.

    See Also
    --------
    DataCache
    MultiprocessedCache
    MultithreadedCache
    """

    if (cache is None) or (not cache):
        return dgen_list

    if (workers is not None) and (workers > 0):
        if concurrency == 'process':
            LOGGER.info(
                "Using multiprocess data generator cache with %d workers",
                workers
            )
            return [
                MultiprocessedCache(x, workers) for x in dgen_list
            ]
        elif concurrency == 'thread':
            LOGGER.info(
                "Using multithreaded data generator cache with %d workers",
                workers
            )
            return [
                MultithreadedCache(x, workers) for x in dgen_list
            ]
        else:
            raise RuntimeError(
                "Unknown concurrency type: %s" % concurrency
            )
    else:
        LOGGER.info("Using data generator cache")
        return [ DataCache(x) for x in dgen_list ]

def add_disk_cache_decorators(dgen_list, use_disk_cache, **kwargs):
    """Add disk cache decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    use_disk_cache : bool
        If True then disk cache decorators will be used. Otherwise, this
        function will return `dgen_list` unmodified.
    **kwargs : dict
        Dictionary that uniquely specifies given disk cache.
        C.f. DataDiskCache constructor.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by `DataDiskCache` decorators

    See Also
    --------
    DataDiskCache
    """

    if not use_disk_cache:
        return dgen_list

    if len(dgen_list) > 2:
        return dgen_list

    LOGGER.info("Using disk based data generator cache")
    return [
        DataDiskCache(dgen = dgen, part = idx, **kwargs)
            for idx,dgen in enumerate(dgen_list)
    ]

def add_weights(dgen_list, batch_size, weights):
    """Add weight decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    batch_size : int
        Size of the batches that DataGenerators from `dgen_list` are going
        to generate.
    weights : dict or str or None
        Weights specification. C.f. `get_weights`. If None then this function
        will return `dgen_list` unmodified.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by `DataWeight` decorators.

    See Also
    --------
    DataWeight
    get_weights
    """

    if weights is None:
        return dgen_list

    return [
        DataWeight(x, batch_size, get_weights(weights)) for x in dgen_list
    ]

def add_prong_sorters(
    dgen_list, prong_sorters, vars_input_png2d, vars_input_png3d
):
    """
    Add prong sorting decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    prong_sorters : dict or None
        Prong sorting specifications to use for 2D and 3D prongs of the form
        { 'input_png2d' : PRONG_SORT_TYPE, 'input_png3d' : PRONG_SORT_TYPE }.
        The parameter PRONG_SORT_TYPE will be passed to a constructor of the
        `DataProngSorter`. If `prong_sorter` is None then this function will
        return `dgen_list` unmodified.
    vars_input_png2d : list of str or None
        Names of 2D prong level input variables that DataGenerators from
        `dgen_list` will be generating batches for.
    vars_input_png3d : list of str or None
        Names of 3D prong level input variables that DataGenerators from
        `dgen_list` will be generating batches for.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by `DataProngSorter`.

    See Also
    --------
    DataProngSorter
    """

    if prong_sorters is None:
        return dgen_list

    for k,v in prong_sorters.items():
        if k == 'input_png2d':
            LOGGER.info("Adding png2d prong sorter: '%s'", k)
            input_vars = vars_input_png2d
        elif k == 'input_png3d':
            LOGGER.info("Adding png3d prong sorter: '%s'", k)
            input_vars = vars_input_png3d
        else:
            raise ValueError("Unknown prong input name '%s'" % (k))

        dgen_list = [
            DataProngSorter(x, v, k, input_vars) for x in dgen_list
        ]

    return dgen_list

def create_basic_data_generators(
    datadir            = None,
    dataset            = None,
    batch_size         = 1024,
    max_prongs         = None,
    seed               = None,
    test_size          = 0.2,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None,
    disk_cache         = None,
):
    """
    Load dataset, shuffle, and create train/test DataGenerators.

    Parameters
    ----------
    datadir : str
        Root directory where datasets are located.
    dataset : str
        Path relative `datadir` to load dataset from.
    batch_size : int
        Size of the batches to be generated.
    max_prongs : int or None, optional
        If `max_prongs` is not None, then the number of 2D and 3D prongs will
        be truncated by `max_prongs`. Default: None.
    seed : int or None
        Seed that will be used to shuffle data.
    test_size : int or float or None
        Amount of samples from `data_loader` that will go to the test sample.
        C.f. `train_test_split`.
    vars_input_slice : list of str or None, optional
        Names of slice level input variables in `data_loader`.
    vars_input_png3d : list of str or None, optional
        Names of 3d prong level input variables in `data_loader`.
    vars_input_png2d : list of str or None, optional
        Names of 2d prong level input variables in `data_loader`.
    var_target_total : str or None, optional
        Name of the variable in `data_loader` that holds total energy of
        the event (e.g. neutrino energy).
    var_target_primary : str or None, optional
        Name of the variable in `data_loader` that holds primary energy of
        the event (e.g. lepton energy).
    disk_cache : bool or None
        If True then disk cache decorators will be used.

    Returns
    -------
    [ DataGenerator, DataGenerator ]
        Train and test DataGenerators.

    See Also
    --------
    construct_data_loader
    DataGenerator
    add_disk_cache_decorators
    """

    LOGGER.info("Loading %s dataset from %s.", dataset, datadir)
    path = os.path.join(datadir, dataset)
    data_loader_list = construct_data_loader(path, seed, test_size)

    LOGGER.info(
          "Creating data generators with:\n"
        + "    batch size   : %d\n" % (batch_size)
        + "    max prongs   : %s\n" % (max_prongs)
        + "    seed         : %s\n" % (seed)
        + "    test size    : %s\n" % (test_size)
    )

    dgen_list = [
        DataGenerator(
            x, batch_size, max_prongs,
            vars_input_slice, vars_input_png3d, vars_input_png2d,
            var_target_total, var_target_primary,
        )
        for x in data_loader_list
    ]

    return add_disk_cache_decorators(
        dgen_list, disk_cache,
        datadir            = datadir,
        dataset            = dataset,
        batch_size         = batch_size,
        max_prongs         = max_prongs,
        vars_input_slice   = vars_input_slice,
        vars_input_png3d   = vars_input_png3d,
        vars_input_png2d   = vars_input_png2d,
        var_target_total   = var_target_total,
        var_target_primary = var_target_primary,
    )

def create_data_generators(
    datadir            = None,
    dataset            = None,
    batch_size         = 1024,
    max_prongs         = None,
    noise              = None,
    prong_sorters      = None,
    seed               = None,
    test_size          = 0.2,
    weights            = None,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None,
    cache              = True,
    disk_cache         = True,
    concurrency        = None,
    workers            = 1,
):
    """
    Construct train/test DataGenerators from a dataset.

    Parameters
    ----------
    datadir : str
        Root directory where datasets are located.
    dataset : str
        Path relative `datadir` to load dataset from.
    batch_size : int
        Size of the batches to be generated.
    max_prongs : int or None, optional
        If `max_prongs` is not None, then the number of 2D and 3D prongs will
        be truncated by `max_prongs`. Default: None.
    noise : list of dict or dict or None
        Noise configuration. C.f. `add_noise`.
    prong_sorters : dict or None
        Prong sorting specifications. C.f. `add_prong_sorters`.
    seed : int or None
        Seed that will be used to shuffle data.
    test_size : int or float or None
        Amount of samples from `data_loader` that will go to the test sample.
        C.f. `train_test_split`.
    weights : dict or str or None
        Weights specification. C.f. `add_weights`.
    vars_input_slice : list of str or None, optional
        Names of slice level input variables in `data_loader`.
    vars_input_png3d : list of str or None, optional
        Names of 3d prong level input variables in `data_loader`.
    vars_input_png2d : list of str or None, optional
        Names of 2d prong level input variables in `data_loader`.
    var_target_total : str or None, optional
        Name of the variable in `data_loader` that holds total energy of
        the event (e.g. neutrino energy).
    var_target_primary : str or None, optional
        Name of the variable in `data_loader` that holds primary energy of
        the event (e.g. lepton energy).
    cache : bool or None
        Specifies whether to cache batches in RAM. C.f. `add_cache_decorators`.
    disk_cache : bool or None
        Specifies whether to cache batches on disk.
        C.f. `add_disk_cache_decorators`.
    concurrency : { None, 'thread', 'process' }
        If not None then batches will be precomputed in parallel.
        C.f. `add_cache_decorators`.
    workers : int or None
        Number of parallel threads/processes to use for precomputing batches.
        C.f. `add_cache_decorators`.

    Returns
    -------
    [ IDataGenerator, IDataGenerator ]
        Train and test DataGenerators that can be used for training with
        `keras`.

    See Also
    --------
    create_basic_data_generators
    add_weights
    add_cache_decorators
    add_prong_sorters
    add_noise
    """


    dgen_list = create_basic_data_generators(
        datadir, dataset, batch_size, max_prongs, seed, test_size,
        vars_input_slice, vars_input_png3d, vars_input_png2d,
        var_target_total, var_target_primary, disk_cache
    )

    dgen_list = add_weights(dgen_list, batch_size, weights)
    dgen_list = add_cache_decorators(dgen_list, cache, concurrency, workers)

    dgen_list = add_prong_sorters(
        dgen_list, prong_sorters, vars_input_png2d, vars_input_png3d
    )
    dgen_list = add_noise(dgen_list, noise)
    dgen_list = [ DataNANMask(x) for x in dgen_list ]

    # pylint: disable = import-outside-toplevel
    from lstm_ee.data.data_generator.keras_sequence import KerasSequence
    dgen_list = [ KerasSequence(x) for x in dgen_list ]

    return dgen_list

def load_data(args):
    """
    Wrapper around `create_data_generators` that unpacks arguments from `args`.
    """

    return create_data_generators(
        datadir            = args.root_datadir,
        dataset            = args.dataset,
        batch_size         = args.batch_size,
        max_prongs         = args.max_prongs,
        noise              = args.noise,
        prong_sorters      = args.prong_sorters,
        seed               = args.seed,
        test_size          = args.test_size,
        weights            = args.weights,
        vars_input_slice   = args.vars_input_slice,
        vars_input_png3d   = args.vars_input_png3d,
        vars_input_png2d   = args.vars_input_png2d,
        var_target_total   = args.var_target_total,
        var_target_primary = args.var_target_primary,
        cache              = args.cache,
        disk_cache         = args.disk_cache,
        concurrency        = args.concurrency,
        workers            = args.workers,
    )


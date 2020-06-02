Data Handling
=============

This page will try to document data formats that `lstm_ee` employs, data
generation performance and ways to improve it.

Data Formats
------------

Currently, `lstm_ee` supports reading data from ``csv`` and ``hdf5`` files.

CSV Files
^^^^^^^^^

`lstm_ee` package support reading data from the ``csv`` files. It supports
reading both plain files and compressed (*gzip*, *xz*, etc) files with
``pandas.read_csv``.

Since `lstm_ee` relies on a prong level variables, that are essentially a
variable length arrays, it needs to store variable length arrays in the ``csv``
files. It was decided to store a variable length arrays serialized as a comma
separated string of values in a ``csv``. So, for example the following
serialized string

::

    "0.2,0.334,0.564,1.4"

will correspond to a variable length array

::

    [ 0.2, 0.334, 0.564, 1.4 ]

CSV Performance
~~~~~~~~~~~~~~~

Serialization of variable length arrays requires to have a custom parser to
deserialize them. Deserializing arrays takes significant time and hits data
handling performance.

Additionally, ``pandas.DataFrame`` is used to handle the raw ``csv`` files.
It loads the entire ``csv`` file into RAM. ``pandas`` has huge memory overhead
for ``pandas.DataFrame``, e.g. if your ``csv`` file has size around 10G, then
the loaded ``pandas.DataFrame`` may occupy up to 40G in RAM.

.. note::
    `lstm_ee` likely needs a custom ``csv`` file parser to avoid enormous
    ``pandas`` memory overhead.

.. note::
    Variable length array deserialization is **the** performance bottleneck in
    the case of ``csv`` files.


HDF5 Files
^^^^^^^^^^

`lstm_ee` also supports reading data from the ``hdf5`` files. However, the
``hdf5`` files are expected to have a certain structure. Namely, all data
arrays should be stored in the root of the file, one array per variable.
Prong variables should be stored as arrays of variable length arrays.

Here is an example of the expected file structure:

::

    test.h5
        /calE, type earray, shape (N,)
        /nHit, type earray, shape (N,)
        ...
        /png.calE, type vlarray, shape (N,)
        /png.nhit, type vlarray, shape (N,)
        ...

Here ``/calE``, ``/hHit`` are datasets of slice level calorimetric energies
and number of hits. And ``/png.calE``, ``/png.hHit`` are datasets of 3D prong
level calorimetric energies and numbers of hits. All datasets in the ``hdf5``
file should have the same shape.

.. note::
    This structure differs from a structure of ``hdf5`` files that NOvA uses.
    NOvA ``hdf5`` files are not supported in `lstm_ee`. The author of `lstm_ee`
    package has an experimental converter from NOvA ``hdf5`` to `lstm_ee`
    ``hdf5`` files, but it is a mess :(

You can convert a ``csv`` file to an ``hdf5`` file by using script
``scripts/data/csv_to_hdf.py``.

HDF5 Performance
~~~~~~~~~~~~~~~~

Compared to ``csv`` files in  ``hdf5`` once does not have to deserialize
variable length arrays. This gives a performance advantage. ``hdf5`` files
also do not have to be loaded into RAM, before one can work with them.

On the downside, ``hdf5`` datasets have terrible random access performance.
In its current implementation `lstm_ee` shuffles dataset indices before
separating them into training/validation parts. Loading data with shuffled
indices takes forever.

.. note::
    This can be fixed by *pre*-shuffling the data and then loading dataset
    in contiguous chunks. Such data handling mode is not supported yet.

.. warning::
    Do not try to solve this simply by disabling data shuffling. There is a
    subtle behavior of the Batch Normalization layers on an unshuffled dataset
    that will result in a significant difference between training/validation
    losses.

    If you do remove shuffling then make sure to use Batch Re-Normalization
    layers and tune them to remove any discrepancy between training and
    validation losses.

Data Generation Performance
---------------------------

Another aspect of data handling that has performance impact is joining
raw data arrays into batches (fixed size ``numpy.ndarray`` s) that will be fed
to the ``keras`` models. It is problematic to do for the prong variables, since
internally they are represented as arrays of variable length arrays.

The author of this package was not able to efficiently implement batching
of variable length arrays into a single fixed size ``numpy.ndarray``.

.. note::
    If you are interested in solving this, you can try optimizing function

    ::

        lstm_ee.data.data_generator.funcs.funcs_varr.join_varr_arrays

    Its partially optimized cython implementation (2.8 times faster than python
    version) is here

    ::

        lstm_ee.data.data_generator.funcs.funcs_varr_opt.c_join_varr_arrays

    but additional optimizations are possible

.. warning::
    Note though that currently the performance bottleneck is not the joining
    data arrays into batches but either deserializing variable length
    arrays in the case of ``csv`` files, or random data access in the case
    of ``hdf5`` files.

Caches and Multiprocessing
--------------------------

The slow generation of data batches may significantly impact `lstm_ee` training
and evaluation speeds. There are a few way to improve the data generation
performance:

1. Cache generated data batches.
2. Use concurrent data generation.

In this section I will give a brief overview of these options.

Data Caches
^^^^^^^^^^^

Since during the training of neural network the same batch of data is reused
multiple times, the simplest way to speed up the training would be to cache
generated data batches. `lstm_ee` package supports two kinds of caches -- RAM
based cache and Disk based cache.

The RAM based cache stores all generated data batches in RAM. This creates
memory overhead (about the size of the original dataset), but makes any
training blazingly fast. To activate the RAM based cache set ``cache`` option
of the training parameters to ``True``.

The Disk based cache stores generated data batches on a disk. Loading generated
batches from a disk is slightly faster than generating them from scratch.
Therefore, while Disk based cache is slower than the RAM cache, unlike RAM
based cache it does not create any RAM overhead and it is persistent between
different trainings. To activate the Disk based cache set ``disk_cache`` option
of the training parameters to ``True``.

Concurrent Data Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to exploit concurrency for data generation. First, is to use
builtin parallelization support in ``keras``. This works fine when it works,
but ``keras`` support of parallel data generation is extremely buggy.
Another way to use concurrency for data generation is to rely on `lstm_ee`
routines to precompute and cache data batches in parallel.

In other words, if your ``keras`` version is able to handle parallel data
generation without issues, then you would probably want to use builtin
``keras`` concurrency. Otherwise, you would have to fall back to the `lstm_ee`
way.

To activate a concurrent data generation you would need to set ``workers``
training option to the number of parallel jobs that will be used for data
generation. You will also need set value of the ``concurrency`` training
parameter to the type of the concurrency model you want to use: *thread* based
concurrency or *process* based concurrency.

By default the ``keras`` builtin parallelization will be used. However, if you
also activate the RAM based cache with ``cache`` training option then the
`lstm_ee` concurrency will be used instead.

.. note::
    The tread based concurrency model is largely useless, since python's GIL
    effectively serializes any concurrency.

.. note::
    The process based concurrency model works fine. However, when multiple
    parallel processes are created they receive a copy of the dataset.
    You may quickly ran out of RAM when launching multiple workers.


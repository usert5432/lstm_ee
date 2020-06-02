"""
Definition of a CSVLoader object for loading datasets from csv files.
"""

import threading

import pandas as pd
import numpy  as np

from .idata_loader import IDataLoader

class CSVLoader(IDataLoader):
    """DataLoader for loading data from the csv files.

    This class is a wrapper around `pandas.DataFrame` to handle special csv
    files that `lstm_ee` relies on.

    The `lstm_ee` uses both slice level and prong level data. The slice level
    data can be easily fit in a `pandas.DataFrame`, since values for each
    variable are just lists of scalars. However, the prong level data is a
    list of variable length arrays (number of prongs varies for each slice).

    To store variable length arrays in the csv files, `lstm_ee` serializes them
    as strings of the form "value0,value1,value2,...". Therefore, `CSVLoader`
    is a wrapper around `pandas.DataFrame` such that when asked to return
    values for a given variable by calling `get` function:
       - if values are numeric -- it will return them unmodified, similar to
         the `pandas.DataFrame`.
       - if on the other hand values are strings, then it will try to
         deserialize them as variable length arrays. The returned value
         will be a numpy array of numpy arrays, with first dimension indexing
         slices and second prongs.

    Parameters
    ----------
    path : str
        Path to the csv file with the dataset.

    Notes
    -----
    `CSVLoader` uses `threading.Lock` for accessing internal `pandas.DataFrame`
    since otherwise access to `pandas.DataFrame` fails in the multithreading
    setting.

    Using `pandas.DataFrame` as a raw csv dataset holder produces HUGE overhead
    in terms of RAM. Likely need more efficient container.

    Using `CSVLoader` for the multiprocessing data generation will result in
    each worker having a separate copy of `CSVLoader` and correspondingly a
    separate copy of the underlying `DataFrame`. This will further exacerbate
    RAM problem.

    Currently, unpacking serialized variable length array is the bottleneck in
    data generation. This needs to be addressed somehow. Until it is addressed
    you should use data generator caches.
    """
    # pylint: disable=no-self-use

    def __init__(self, path):
        super(CSVLoader, self).__init__()

        self._df = pd.read_csv(path)

        self._fname     = path
        self._variables = list(self._df.columns)
        self._len       = len(self._df)
        self._lock      = threading.Lock()

    def variables(self):
        return self._variables

    def _lazy_load(self):
        if self._df is None:
            self._df = pd.read_csv(self._fname)

        if self._lock is None:
            self._lock = threading.Lock()

    def __getstate__(self):
        """Serialize object for pickle.

        This function is called when `CSVLoader` is serialized by `pickle`.

        Notes
        -----
        Pickling is required for multiprocessing.

        Pickling the entire `pandas.DataFrame` that `CSVLoader` holds is
        inefficient, and does not always work (sometimes it is too large to be
        pickled). Therefore, when pickling we first drop the `DataFrame` and
        reload it later lazily at first use.

        `threading.Lock` that `CSVLoader` is using cannot be pickled. So we
        also drop it and create when it is used.
        """

        if self._df is not None:
            self._df = None

        if self._lock is not None:
            self._lock = None

        return self.__dict__

    def _convert_varr_series(self, s, dtype):
        """Deserialize a sequence of variable length arrays"""

        def convert_varrstr_to_varr(x, dtype):
            if pd.isnull(x):
                return np.empty((0,), dtype = dtype)

            if isinstance(x, str):
                return np.array(
                    [ float(y) for y in x.split(',') ], dtype = dtype
                )

            return np.array([ x ], dtype = dtype)

        return s.apply(
            lambda x, dtype = dtype : convert_varrstr_to_varr(x, dtype)
        )

    def get(self, var, index = None):
        self._lazy_load()

        if isinstance(var, list):
            if len(var) != 1:
                raise RuntimeError("Invalid var: %s" % var)
            var = var[0]

        with self._lock:
            if index is None:
                s = pd.Series(self._df.loc[:, var])
            else:
                s = self._df.loc[index, var]

        if np.issubdtype(s.dtype, np.number):
            result = s
        else:
            result = self._convert_varr_series(s, np.float32)

        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.values

        return result

    def __len__(self):
        return self._len


"""
Functions for working with batches of variable length arrays.
"""

import numpy as np

import pyximport
pyximport.install(
    language_level = 3,
    setup_args     = { "include_dirs" : [ np.get_include() ] }
)

# pylint: disable=import-error,wrong-import-position
from .funcs_varr_opt import c_join_varr_arrays

def sort_unpacked_varr_arrays(unpacked_array, sort_var_idx, ascending = False):
    """Sort batch of variable length array variables inplace.

    Parameters
    ----------
    unpacked_array : ndarray, shape (N_SAMPLE, N_VARR, N_VAR)
        A batch of variable length arrays joined into a single `np.ndarray` to
        be sorted along second axis.
    sort_var_idx : int
        Index of the variable in the third axis of `unpacked_array` which
        values will be used for determining order along the second axis.
    ascending : bool, optional
        If True it will sort second dimension in ascending order, otherwise
        in descending order. Default: False.
    """

    for row_idx in range(unpacked_array.shape[0]):
        values = unpacked_array[row_idx, :, sort_var_idx]
        values = values[~np.isnan(values)]

        if ascending:
            indices = values.argsort()
        else:
            indices = (-values).argsort()

        unpacked_array[row_idx, :len(indices), :] = \
            unpacked_array[row_idx, indices, :]
        unpacked_array[row_idx, len(indices):, :] = np.nan

def shuffle_unpacked_varr_arrays(unpacked_array):
    """Shuffle batch of variable length array variables inplace.

    Parameters
    ----------
    unpacked_array : ndarray, shape (N_SAMPLE, N_VARR, N_VAR)
        A batch of variable length arrays joined into a single `np.ndarray` to
        be sorted along second axis.
    """

    idx_template = np.arange(unpacked_array.shape[1])

    for row_idx in range(unpacked_array.shape[0]):
        indices = idx_template.copy()[~np.isnan(unpacked_array[row_idx, :, 0])]
        np.random.shuffle(indices)

        unpacked_array[row_idx, :len(indices), :] = \
            unpacked_array[row_idx, indices, :]
        unpacked_array[row_idx, len(indices):, :] = np.nan

def join_varr_arrays(raw_varr_list, length_limit = None):
    """Join a list of variable length arrays batches into a `np.ndarray`.

    This function joins variable length array batches from `raw_varr_list` into
    a single fixed size `np.ndarray` by padding (using NaNs) all variable
    length arrays to a common size.

    Each element of `raw_varr_list` is supposed to contain a batch of variable
    length arrays for a given variable (N_VAR = len(`raw_varr_list`)).

    All elements of `raw_varr_list` should have the same length N_SAMPLE.

    Each element of `raw_varr_list` is supposed to be a `np.ndarray` of
    variable length arrays (also `np.ndarray`). That, is a numpy array
    of numpy arrays.

    All batches of variable length arrays from `raw_varr_list` will be joined
    into a `np.ndarray` of the shape (N_SAMPLE, N_VARR, N_VAR), where second
    axis will be the axis along the variable length dimension.

    Parameters
    ----------
    raw_varr_list : list of ndarray of ndarray
        List of variable length array batches to be joined together.
    length_limit : int or None, optional
        if `length_limit` is not None it will limit lengths of variable length
        arrays by `length_limit`. Otherwise, the dimension along N_VARR axis
        will be determined as a maximum of all variable length dimensions of
        `raw_varr_list`.

    Return
    ------
    ndarray, shape (N_SAMPLE, N_VARR, N_VAR)
        Joined batches of variable length arrays.

    Notes
    -----
    This function is awfully slow. It is the bottleneck of the data batch
    generation. It must be rewritten in a more efficient matter.

    C.f. cython version `c_join_varr_arrays` that is around 2.4 times faster,
    but still slow.
    """

    n_var = len(raw_varr_list)
    if n_var == 0:
        return None

    n_row = len(raw_varr_list[0])
    n_png = max([ len(x) for x in raw_varr_list[0] ])

    if length_limit is not None:
        n_png = min(n_png, length_limit)

    # result (row_idx, varr_idx, var_idx)
    result = np.full((n_row, n_png, n_var), np.nan)

    for var_idx in range(n_var):
        values = raw_varr_list[var_idx]

        for row_idx in range(n_row):
            row_values = values[row_idx]
            row_n_png = min(n_png, len(row_values))

            result[row_idx, :row_n_png, var_idx] = row_values[:row_n_png]

    return result

def unpack_varr_arrays(data_loader, variables, index, length_limit = None):
    """Unpack variable length arrays from data_loader into a `np.ndarray`.

    This function extracts a variable length arrays variables from the
    `data_loader` and unpacks them into a fixed size `np.ndarray`, suitable
    for feeding into ML frameworks. During unpacking all missing variable
    length arrays values are padded by NaNs.

    The returned `np.ndarray` will have shape (N_SAMPLE, N_VARR, N_VAR).
    The first axis is the returned array is the axis along `data_loader`
    dimension. The third axis of the returned array is the axis along the
    `variables` list. The second axis of the returned array is the axis along
    the variable length arrays dimension.

    Parameters
    ----------
    data_loader : IDataLoader
        `IDataLoader` that holds values of `variables`
    variables : list of str
        List of variable names of variable length arrays to be unpacked.
    index : int or ndarray or None
        Index of values to be retrieved from `data_loader`.
        If None, all values from `data_loader` will be used.
    length_limit : int or None, optional
        If 'length_limit' is not None, the variable length arrays will be
        truncated by `length_limit`.

    Return
    ------
    ndarray, shape (N_SAMPLE, N_VARR, N_VAR)
        Joined batches of variable length arrays.

    See Also
    --------
    join_varr_arrays
    """

    return c_join_varr_arrays(
        [ data_loader.get(v, index) for v in variables ], length_limit
    )


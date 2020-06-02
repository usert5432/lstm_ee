#cython: infer_types=True
#cython: profile=False
#cython: linetrace=False
#cython: nonecheck=False
#cython: initializedcheck=False

cimport cython

import  numpy as np
cimport numpy as cnp

from libc.math cimport NAN

ctypedef cnp.float32_t CTYPE
DTYPE = np.float32

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t get_varr_size(varr, length_limit = None):
    cdef Py_ssize_t result = 0
    cdef Py_ssize_t n_rows = varr.shape[0]

    cdef Py_ssize_t row_idx

    for row_idx in range(n_rows):
        result = max(result, varr[row_idx].shape[0])

    if length_limit is not None:
        result = min(result, length_limit)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def c_join_varr_arrays(list raw_varr_list, length_limit = None):

    cdef Py_ssize_t row_idx
    cdef Py_ssize_t png_idx
    cdef Py_ssize_t var_idx

    cdef Py_ssize_t n_rows = 0
    cdef Py_ssize_t n_pngs = 0
    cdef Py_ssize_t n_vars = 0

    cdef Py_ssize_t row_n_pngs = 0

    n_vars = len(raw_varr_list)
    if n_vars == 0:
        return None

    n_rows = raw_varr_list[0].shape[0]
    n_pngs = get_varr_size(raw_varr_list[0], length_limit)

    cdef cnp.ndarray[CTYPE, ndim=3] result = np.empty(
        (n_rows, n_pngs, n_vars), dtype = DTYPE
    )
    #cdef cnp.ndarray[CTYPE, ndim=1] row_values

    for var_idx in range(n_vars):
        var_values = raw_varr_list[var_idx]

        for row_idx in range(n_rows):
            row_values = var_values[row_idx]
            row_n_pngs = min(n_pngs, row_values.shape[0])

            for png_idx in range(row_n_pngs):
                result[row_idx, png_idx, var_idx] = row_values[png_idx]

            for png_idx in range(row_n_pngs, n_pngs):
                result[row_idx, png_idx, var_idx] = NAN

    return result


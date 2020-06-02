"""
Classes for changing prong order
"""

import re
from .funcs_varr import sort_unpacked_varr_arrays, shuffle_unpacked_varr_arrays

class ProngSorter:
    """Basic interface for ProngSorter class

       The `ProngSorter` classes should __call__ function that will receive
       an array of prong variables and is expected to sort them inplace.
    """

    def __call__(self, values):
        """Sort prongs inplace

        Parameters
        ----------
        values : ndarray, shape (N_SAMPLE, N_PRONG, N_VAR)
            Prong values to be sorted.
        """
        raise NotImplementedError

class SingleVarProngSorter(ProngSorter):
    """`ProngSorter` that sorts prongs by values of single variable.

    Parameters
    ----------
    var_name : str
        Name of variable according to which prongs will be sorted prefixed
        by a character ('+' or '-') indicating prong order. For example,
        "+var_name" will sort prongs by 'var_name` in the ascending (highest
        values first) order, while "-var_name" will sort prongs in the
        descending (lowest values first) order.
    prong_var_list : list of str
        List of prong variables. The `SingleVarProngSorter` needs to know
        names of input variables whose values it will be sorting later.
    """

    def __init__(self, var_name, prong_vars_list):

        regexp = re.compile(r'([+-])(.*)')
        res    = regexp.match(var_name)
        if not res:
            raise ValueError('Failed to parse prong sorter %s' % var_name)

        self._asc     = (res.group(1) == '+')
        self._var     = res.group(2)
        self._var_idx = prong_vars_list.index(self._var)

        assert(self._var_idx >= 0)

    def __call__(self, unpacked_prong_array):
        sort_unpacked_varr_arrays(
            unpacked_prong_array, self._var_idx, self._asc
        )

class RandomizedProngSorter(ProngSorter):
    """`ProngSorter` that randomizes prong order."""

    def __call__(self, unpacked_prong_array):
        shuffle_unpacked_varr_arrays(unpacked_prong_array)


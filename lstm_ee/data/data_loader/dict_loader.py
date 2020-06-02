"""
Definition of a DictLoader that constructs IDataLoader from a dictionary.
"""

import numpy as np
from .idata_loader import IDataLoader

class DictLoader(IDataLoader):
    """DataLoader constructed from a dictionary.

    This object creates a DataLoader from a dictionary of supplied by the user.
    Mostly useful for debugging.

    Parameters
    ----------
    data_dict : dict
        Dictionary of the form { 'variable name' : values }. values can be
        either:
          - list of numbers -- will be treated as scalar values.
          - list of of list of numbers -- will be treated as a list of variable
            length arrays.
    """

    def __init__(self, data_dict):
        super(DictLoader, self).__init__()

        self._variables = list(data_dict.keys())

        if self._variables:
            self._len = len(data_dict[self._variables[0]])
        else:
            self._len = 0

        self._prepare_dict(data_dict)

    def _prepare_dict(self, data_dict):
        """Convert values from `data_dict` into proper numpy arrays."""
        self._dict = {}

        for var in self._variables:
            values = data_dict[var]

            # pylint: disable=len-as-condition
            # since `values` can be a numpy array for which idiom
            # 'if values' does not work
            if len(values) > 0:
                if isinstance(values[0], (list, np.ndarray)):
                    # Case of varr values
                    self._dict[var] = np.array([ np.array(x) for x in values ])
                else:
                    # Case of scalar values
                    self._dict[var] = np.array(values)
            else:
                self._dict[var] = np.array([])

    def __len__(self):
        return self._len

    def variables(self):
        return self._variables

    def get(self, var, index = None):
        if index is None:
            return self._dict[var]
        else:
            return self._dict[var][index]


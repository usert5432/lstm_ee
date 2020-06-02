"""
Definition of a DataLoader Interface.
"""

class IDataLoader():
    """An interface for DataLoader object.

    A DataLoader is an object that contains a dataset. A dataset is a
    collection of items { 'variable_name' : values }.

    Values for a given variable can be retrieved from the DataLoader by
    using `get` method. The list of variables that DataLoader holds can
    be retrieved by the `variables` function.

    Values for all variables in a DataLoader should have the same length.
    This length can be determined by calling `len` function on a DataLoader.

    By its behavior DataLoader is similar to the `pandas.DataFrame`, but has
    more abstract and less restrictive API.
    """
    # pylint: disable=no-self-use
    def __init__(self):
        pass

    def variables(self):
        """List variable names that this DataLoader holds"""
        raise NotImplementedError

    def get(self, var, index = None):
        """Return values for a variable `var`

        Parameters
        ----------
        var : str
            Name of the variable to retrieve values for.
        index : int or ndarray or None
            If `index` is None this function will return all values for the
            variable `var`.
            Otherwise, it will return only values specified by `index`.

        Returns
        -------
        ndarray
            Values for variable `var` with index `index`.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



"""
Definition of a base class for decorator pattern around `IDataLoader`.
"""

from .idata_loader import IDataLoader

class IDataLoaderDecorator(IDataLoader):
    """A base class for a decorator around `IDataLoader`."""

    def __init__(self, data_loader):
        super(IDataLoaderDecorator, self).__init__()
        self._data_loader = data_loader

    def variables(self):
        return self._data_loader.variables()

    def get(self, var, index = None):
        return self._data_loader.get(var, index)

    def __len__(self):
        return len(self._data_loader)


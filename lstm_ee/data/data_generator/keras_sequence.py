"""
Definition of a decorator that makes IDataGenerator descendant of `Sequence`.
"""

from keras.utils import Sequence
from .idata_decorator import IDataDecorator

class KerasSequence(IDataDecorator, Sequence):
    """A decorator that makes `IDataGenerator` a descendant `Sequence`.

    The `keras` function `model.fit` expects to receive a data generator that
    inherits from `keras.utils.Sequence` object. This decorator simply inherits
    from the `Sequence` in order to make `keras` work with `IDataGenerator`
    that it decorates.

    Parameters
    ----------
    dgen : IDataGenerator
        `IDataGenerator` to be decorated.
    """

    def __init__(self, dgen):
        IDataDecorator.__init__(self, dgen)

    def __len__(self):
        return len(self._dgen)

    def __getitem__(self, index):
        return self._dgen[index]


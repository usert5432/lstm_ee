"""
Definition of an interface of a decorator around `IDataGenerator`.
"""

from .idata_generator import IDataGenerator

class IDataDecorator(IDataGenerator):
    """An interface to a decorator around `IDataGenerator`"""

    def __init__(self, dgen):
        super(IDataDecorator, self).__init__()
        self._dgen = dgen

    def __len__(self):
        return len(self._dgen)

    def __getitem__(self, index):
        return self._dgen[index]

    @property
    def vars_input_slice(self):
        return self._dgen.vars_input_slice

    @property
    def vars_input_png2d(self):
        return self._dgen.vars_input_png2d

    @property
    def vars_input_png3d(self):
        return self._dgen.vars_input_png3d

    @property
    def var_target_total(self):
        return self._dgen.var_target_total

    @property
    def var_target_primary(self):
        return self._dgen.var_target_primary

    @property
    def data_loader(self):
        return self._dgen.data_loader

    @property
    def weights(self):
        return self._dgen.weights


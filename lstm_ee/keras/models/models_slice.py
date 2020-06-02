"""
Functions to construct models that work only on slice level inputs.
"""

from keras.models import Model
from .funcs import (get_inputs, get_outputs)

def model_slice_linear(
    reg                = None,
    max_prongs         = None,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None
):
    """Construct linear model that uses only slice level inputs. """
    assert(vars_input_png2d is None)
    assert(vars_input_png3d is None)
    assert(reg is None)

    inputs = get_inputs(vars_input_slice, None, None, max_prongs)

    assert(len(inputs) == 1)

    input_slice = inputs[0]
    outputs     = get_outputs(
        var_target_total, var_target_primary, reg, input_slice
    )

    return Model(inputs = inputs, outputs = outputs)


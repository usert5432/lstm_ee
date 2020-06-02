"""
Functions to construct models that work on flattened prong arrays.
"""

from keras.layers import Concatenate, Flatten
from keras.models import Model

from .funcs import modify_layer, add_hidden_layers, get_inputs, get_outputs

def flattened_model(
    max_prongs         = 5,
    reg                = None,
    batchnorm          = False,
    dropout            = None,
    layer_sizes        = None,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None
):
    """Construct fully connected model that flattens prong arrays"""
    assert(max_prongs is not None)
    assert(vars_input_png2d is None)

    layer_sizes = layer_sizes or []

    inputs = get_inputs(
        vars_input_slice, vars_input_png3d, None, max_prongs
    )

    # pylint: disable=unbalanced-tuple-unpacking
    input_slc, input_png = inputs

    input_png_flat = Flatten()(input_png)

    layer_merged = Concatenate()([ input_png_flat, input_slc ])
    layer_merged = modify_layer(
        layer_merged, 'layer_merged', batchnorm
    )

    layer_hidden = add_hidden_layers(
        layer_merged,
        layer_sizes,
        "hidden",
        batchnorm,
        dropout,
        activation         = 'relu',
        kernel_regularizer = reg,
    )

    outputs = get_outputs(
        var_target_total, var_target_primary, reg, layer_hidden
    )

    return Model(inputs = inputs, outputs = outputs)



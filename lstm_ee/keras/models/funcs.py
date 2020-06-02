"""Functions to construct blocks of layers for `lstm_ee`"""

from keras.layers import (
    Activation, Add, BatchNormalization, Bidirectional,
    Dense, Dropout, Input, LSTM, Masking, TimeDistributed
)

from lstm_ee.consts import DEF_MASK

def modify_layer(layer, name, batchnorm = False, dropout = None):
    """Add BatchNorm and/or Dropout on top of `layer`"""

    if dropout is not None:
        name = "%s-dropout" % (name)
        layer = Dropout(dropout, name = name)(layer)

    if batchnorm:
        name = "%s-batchnorm" % (name)
        layer = BatchNormalization(name = name)(layer)

    return layer

def modify_series_layer(
    layer, name, mask = False, batchnorm = False, dropout = None
):
    """Add Mask and/or BatchNorm and/or Dropout on top of series `layer`"""
    if mask:
        name = '%s-masked' % (name)
        layer = Masking(mask_value = DEF_MASK, name = name)(layer)

    if dropout is not None:
        name = "%s-dropout" % (name)
        layer = TimeDistributed(Dropout(dropout), name = name)(layer)

    if batchnorm:
        name = "%s-batchnorm" % (name)
        layer = TimeDistributed(BatchNormalization(), name = name)(layer)
        #layer = BatchNormalization(name = name)(layer)

    return layer

def get_inputs(
    vars_input_slice, vars_input_png3d, vars_input_png2d, max_prongs
):
    """Construct standard `lstm_ee` input layers."""
    inputs = []

    if vars_input_slice is not None:
        input_slice = Input(
            shape = (len(vars_input_slice),),
            dtype = 'float32',
            name  = 'input_slice'
        )
        inputs.append(input_slice)

    if vars_input_png3d is not None:
        input_png3d = Input(
            shape = (max_prongs, len(vars_input_png3d)),
            dtype = 'float32',
            name  = 'input_png3d'
        )
        inputs.append(input_png3d)

    if vars_input_png2d is not None:
        input_png2d = Input(
            shape = (None, len(vars_input_png2d)),
            dtype = 'float32',
            name  = 'input_png2d'
        )
        inputs.append(input_png2d)

    return inputs

def get_outputs(var_target_total, var_target_primary, reg, layer):
    """Construct standard `lstm_ee` output layers."""

    outputs = []

    if var_target_total is not None:
        target_total = Dense(
            1, name = 'target_total', kernel_regularizer = reg
        )(layer)

        outputs.append(target_total)

    if var_target_primary is not None:
        target_primary = Dense(
            1, name = 'target_primary', kernel_regularizer = reg
        )(layer)

        outputs.append(target_primary)

    return outputs

def add_resblock(layer_input, name_prefix, **kwargs):
    """Add a fully connected residual block on top of `layer_input`"""
    input_shape = layer_input.output_shape[1]

    layer_res_fc_1 = Dense(
        input_shape,
        name       = "%s-res-fc1" % (name_prefix),
        activation = 'relu',
        **kwargs
    )(layer_input)

    layer_res_bn_1 = BatchNormalization(
        name = "%s-res-bn1" % (name_prefix)
    )(layer_res_fc_1)

    layer_res_fc_2 = Dense(
        input_shape,
        name       = "%s-res-fc2" % (name_prefix),
        activation = None,
        **kwargs
    )(layer_res_bn_1)

    layer_res_bn_2 = BatchNormalization(
        name = "%s-res-bn2" % (name_prefix),
    )(layer_res_fc_2)

    layer_res_add = Add(
        name = "%s-res-add" % (name_prefix),
    )([layer_res_bn_2, layer_input])

    layer_res_add_act = Activation(
        activation = 'relu',
        name = "%s-res-add-act" % (name_prefix),
    )(layer_res_add)

    return layer_res_add_act

def add_resblocks(layer_input, n, name_prefix, **kwargs):
    """Add `n` fully connected residual blocks on top of `layer_input`"""
    layer = layer_input

    for i in range(n):
        name = "%s-resblock-%d" % (name_prefix, i + 1)
        layer = add_resblock(layer, name, **kwargs)

    return layer

def add_hidden_layers(
    layer_input, layer_sizes, name_prefix, batchnorm, dropout, **kwargs
):
    """Add fully connected layers on top of `layer_input`

    Parameters
    ----------
    layer_input : keras.Layer
        Layer on top of which new layers will be added.
    layer_sizes : list of int
        Shapes of layers to be added
    name_prefix : str
        Prefix to be added to names of new layers.
    batchnorm : bool
        Whether to add BatchNorm on top of FC layers.
    dropout : float or None
        If not None then Dropout layers will be added on top of FC layers
        with a values of dropout of `dropout`.
    kwargs : dict
        Arguments to be passed to the Dense layers constructors.

    Note
    ----
    Do not use `batchnorm` with `dropout` unless you want to be disappointed.

    Returns
    -------
    keras.Layer
        Last layer added on top of `layer_input`
    """
    layer_hidden = layer_input

    for idx,size in enumerate(layer_sizes):
        name = "%s-%d" % (name_prefix, idx + 1)
        layer_hidden = Dense(size, name = name, **kwargs)(layer_hidden)
        layer_hidden = modify_layer(layer_hidden, name, batchnorm, dropout)

    return layer_hidden

def add_hidden_series_layers(
    layer_input, layer_sizes, name_prefix, batchnorm, dropout, **kwargs
):
    """Add fully connected layers on top of series layer `layer_input`
    C.f. `add_hidden_layers` for the description of arguments.

    See Also
    --------
    add_hidden_layers
    """
    layer_hidden = layer_input

    for idx,size in enumerate(layer_sizes):
        name = "%s-%d" % (name_prefix, idx + 1)

        layer_hidden = TimeDistributed(
            Dense(size, **kwargs), name = name
        )(layer_hidden)

        layer_hidden = modify_series_layer(
            layer_hidden, name,
            mask      = False,
            batchnorm = batchnorm,
            dropout   = dropout
        )

    return layer_hidden

def add_stack_of_lstms(
    layer_input, layer_size_dir_pairs, name_prefix, batchnorm, dropout,
    **kwargs
):
    """Add a stack of LSTM layers on top of series layer `layer_input`


    Parameters
    ----------
    layer_input : keras.Layer
        Series layer on top of which new LSTM layers will be added.
    layer_size_dir_pairs : list of (int, str)
        List of pairs that specify number of units and directions of LSTM
        layers to be added.
        Direction can be either 'forward', 'backward' or 'bidirectional'.
    name_prefix : str
        Prefix to be added to names of new LSTM layers.
    batchnorm : bool
        Whether to add BatchNorm on top of the last LSTM layer.
    dropout : float or None
        If not None then Dropout layer will be added on top of the last LSTM
        layer with a value of dropout `dropout`.
    kwargs : dict
        Arguments to be passed to the LSTM layers constructors.

    Returns
    -------
    keras.Layer
        Last layer added on top of `layer_input`
    """

    layer_lstm      = layer_input
    is_middle_layer = True

    for idx,size_dir_pair in enumerate(layer_size_dir_pairs):
        name = "%s-%d" % (name_prefix, idx + 1)

        size, direction = size_dir_pair

        if idx == len(layer_size_dir_pairs) - 1:
            is_middle_layer = False

        if direction == 'forward':
            layer_lstm = LSTM(
                size, name = name, return_sequences = is_middle_layer, **kwargs
            )(layer_lstm)

        elif direction == 'backward':
            layer_lstm = LSTM(
                size, name = name, return_sequences = is_middle_layer,
                go_backwards = True, **kwargs
            )(layer_lstm)

        elif direction == 'bidirectional':
            layer_lstm = Bidirectional(
                LSTM(size, return_sequences = is_middle_layer, **kwargs),
                name = name, merge_mode = 'concat',
            )(layer_lstm)
        else:
            raise ValueError("Unknown direction: '%s'" % [direction])

        layer_lstm = modify_series_layer(
            layer_lstm,
            name      = name,
            mask      = False,
            batchnorm = batchnorm and is_middle_layer,
            dropout   = dropout if is_middle_layer else None
        )

    return layer_lstm


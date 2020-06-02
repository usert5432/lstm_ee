"""
Functions to construct models that use LSTM layers to process prong inputs.
"""

from keras.layers import LSTM, Concatenate
from keras.models import Model

from .funcs import (
    get_inputs, get_outputs, modify_layer, modify_series_layer,
    add_hidden_layers, add_hidden_series_layers, add_resblocks,
    add_stack_of_lstms
)

def make_standard_lstm_branch(
    branch_label, input_layer, hidden_layers_spec, lstm_units, batchnorm,
    dropout, reg, lstm_kwargs = None
):
    """Create the default block of layers to process sequential inputs.

    Parameters
    ----------
    branch_label : str
        Suffix that will be added to layer names.
    input_layer : keras.Layer
        Layer on top of which new will be added.
    hidden_layers_spec : list of int
        List of Dense layer sizes that will be used to preprocess inputs before
        feeding them to the LSTM layer.
    lstm_units : int
        Number of units that the LSTM layer will have.
    batchnorm : bool or None
        If True then the BatchNorm layers will be used to normalize
        activations.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    reg : keras.Regularizer or None
        `keras` regularizer to use.
    lstm_kwargs : dict or None
        Additional arguments to be passed to the LSTM layer constructor.

    Returns
    -------
    keras.Layer
        Last layer that was added on top of `input_layer`

    See Also
    --------
    make_stacked_lstm_branch
    add_hidden_series_layers
    """

    lstm_kwargs = lstm_kwargs or {}

    input_layer = modify_series_layer(
        input_layer, 'input_%s' % (branch_label),
        mask = True, batchnorm = batchnorm
    )

    layer_hidden_pre = add_hidden_series_layers(
        input_layer, hidden_layers_spec, "hidden_pre_%s" % (branch_label),
        batchnorm, dropout,
        activation         = 'relu',
        kernel_regularizer = reg,
    )

    layer_lstm = LSTM(
        lstm_units, kernel_regularizer = reg, recurrent_regularizer = reg,
        **lstm_kwargs
    )(layer_hidden_pre)

    return layer_lstm

def make_stacked_lstm_branch(
    branch_label, input_layer, hidden_layers_spec, lstm_spec, batchnorm,
    dropout, reg, lstm_kwargs = None
):
    """Create a stack of LSTMs to process sequential inputs.

    Parameters
    ----------
    branch_label : str
        Suffix that will be added to layer names.
    input_layer : keras.Layer
        Layer on top of which new will be added.
    hidden_layers_spec : list of int
        List of Dense layer sizes that will be used to preprocess inputs before
        feeding them to the LSTM layer.
    lstm_spec : list of (int, str)
        List of pairs that specify number of units and directions of LSTM
        layers to be added. C.f. `add_stack_of_lstms`
    batchnorm : bool or None
        If True then the BatchNorm layers will be used to normalize
        activations.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    reg : keras.Regularizer or None
        `keras` regularizer to use.
    lstm_kwargs : dict or None
        Additional arguments to be passed to the LSTM layers constructors.

    Returns
    -------
    keras.Layer
        Last layer that was added on top of `input_layer`

    See Also
    --------
    make_standard_lstm_branch
    add_stack_of_lstms
    """

    lstm_kwargs = lstm_kwargs or {}

    input_layer = modify_series_layer(
        input_layer, 'input_%s' % (branch_label),
        mask = True, batchnorm = batchnorm
    )

    layer_hidden_pre = add_hidden_series_layers(
        input_layer, hidden_layers_spec, "hidden_pre_%s" % (branch_label),
        batchnorm, dropout,
        activation         = 'relu',
        kernel_regularizer = reg,
    )

    layer_lstm = add_stack_of_lstms(
        layer_hidden_pre, lstm_spec,
        'lstm_%s' % (branch_label), batchnorm, dropout,
        recurrent_regularizer = reg, kernel_regularizer = reg,
        **lstm_kwargs
    )

    return layer_lstm

def make_standard_postprocess_branch(
    input_layer, hidden_layers_spec, batchnorm, dropout, reg, n_resblocks
):
    """Create the default postprocessing block of layers.

    Parameters
    ----------
    input_layer : keras.Layer
        Layer on top of which new will be added.
    hidden_layers_spec : list of int
        List of Dense layer sizes that will be used to postprocess LSTM layer
        outputs.
    batchnorm : bool or None
        If True then the BatchNorm layers will be used to normalize
        activations.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    reg : keras.Regularizer or None
        `keras` regularizer to use.
    n_resblocks : int or None
        Number of Dense residual block layers to be added after the last Dense
        layer.

    Returns
    -------
    keras.Layer
        Last layer that was added on top of `input_layer`

    See Also
    --------
    make_standard_lstm_branch
    make_stacked_lstm_branch
    add_hidden_layers
    add_resblocks
    """


    layer_hidden_post = add_hidden_layers(
        input_layer, hidden_layers_spec, "hidden_post", batchnorm, dropout,
        activation = 'relu', kernel_regularizer = reg,
    )

    layer_resnet = add_resblocks(
        layer_hidden_post, n_resblocks, 'resblocks', kernel_regularizer = reg
    )

    return layer_resnet

def model_lstm_v1(
    lstm_units         = 16,
    max_prongs         = 5,
    reg                = None,
    batchnorm          = False,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None
):
    """Create version 1 LSTM network.

    This is the vanilla network that Alexander Radovic used.
    This network uses only 3D prong and slice level inputs and limits the
    number of prongs to 5.
    No input preprocessing or output postprocessing is done.

    Parameters
    ----------
    lstm_units : int, optional
        Number of units that LSTM layer will have. Default: 16.
    max_prongs : int or None, optional
        Limit on the number of prongs that will be used. Default: 5.
    reg : keras.Regularizer or None, optional
        Regularization to use. Default: None
    batchnorm : bool or None, optional
        Whether to use Batch Normalization. Default: False.
    vars_input_slice : list of str or None
        List of slice level input variable names.
    vars_input_png3d : list of str or None
        List of 3D prong level input variable names.
    vars_input_png2d : None
        List of 2D prong level input variable names.
        This is dummy input variable and MUST be None.
    var_target_total : str or None
        Name of the variable that defines true total energy.
    var_target_primary : str or None
        Name of the variable that defines true primary energy.

    Returns
    -------
    keras.Model
        Model that defines the network.

    See Also
    --------
    model_lstm_v2
    model_lstm_v3
    """
    assert(vars_input_png2d is None)

    inputs = get_inputs(vars_input_slice, vars_input_png3d, None, max_prongs)
    # pylint: disable=unbalanced-tuple-unpacking
    input_slc, input_png = inputs

    input_png = modify_series_layer(
        input_png, 'input_png', mask = True, batchnorm = batchnorm
    )

    layer_png_1 = LSTM(
        lstm_units, kernel_regularizer = reg, recurrent_regularizer = reg,
    )(input_png)

    layer_merged = Concatenate()([ layer_png_1, input_slc ])
    layer_merged = modify_layer(layer_merged, 'layer_merged', batchnorm)

    outputs = get_outputs(
        var_target_total, var_target_primary, reg, layer_merged
    )

    model = Model(inputs = inputs, outputs = outputs)

    return model

def model_lstm_v2(
    lstm_units         = 16,
    layers_pre         = [],
    layers_post        = [],
    n_resblocks        = None,
    max_prongs         = 5,
    reg                = None,
    batchnorm          = False,
    dropout            = None,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None
):
    """Create version 2 LSTM network.

    This is a modification of the vanilla network that Alexander Radovic used.
    This network also uses only 3D prong and slice level inputs.
    However, it does LSTM input preprocessing and output postprocessing.

    Parameters
    ----------
    lstm_units : int, optional
        Number of units that LSTM layer will have. Default: 16.
    layers_pre : list of int
        List of Dense layer sizes that will be used to preprocess prong inputs.
    layers_post : list of int
        List of Dense layer sizes that will be used to postprocess LSTM
        outputs.
    n_resblocks : int or None, optional
        Number of the fully connected residual blocks to be added before the
        output layer. Default: None
    max_prongs : int or None, optional
        Limit on the number of prongs that will be used. Default: 5.
    reg : keras.Regularizer or None, optional
        Regularization to use. Default: None
    batchnorm : bool or None, optional
        Whether to use Batch Normalization. Default: False.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    vars_input_slice : list of str or None
        List of slice level input variable names.
    vars_input_png3d : list of str or None
        List of 3D prong level input variable names.
    vars_input_png2d : None
        List of 2D prong level input variable names.
        This is dummy input variable and MUST be None.
    var_target_total : str or None
        Name of the variable that defines true total energy.
    var_target_primary : str or None
        Name of the variable that defines true primary energy.

    Returns
    -------
    keras.Model
        Model that defines the network.

    See Also
    --------
    model_lstm_v1
    model_lstm_v3
    """

    # pylint: disable=dangerous-default-value
    assert(vars_input_png2d is None)

    inputs = get_inputs(
        vars_input_slice, vars_input_png3d, vars_input_png2d, max_prongs
    )
    # pylint: disable=unbalanced-tuple-unpacking
    input_slc, input_png = inputs

    layer_png_1 = make_standard_lstm_branch(
        'png', input_png, layers_pre, lstm_units, batchnorm, dropout, reg
    )

    layer_merged = Concatenate()([ layer_png_1, input_slc ])
    layer_merged = modify_layer(layer_merged, 'layer_merged', batchnorm)
    layer_post   = make_standard_postprocess_branch(
        layer_merged, layers_post, batchnorm, dropout, reg, n_resblocks
    )

    outputs = get_outputs(
        var_target_total, var_target_primary, reg, layer_post
    )

    return Model(inputs = inputs, outputs = outputs)

def model_lstm_v3(
    lstm_units3d       = 16,
    lstm_units2d       = 16,
    layers_pre         = [],
    layers_post        = [],
    n_resblocks        = 0,
    max_prongs         = None,
    reg                = None,
    batchnorm          = False,
    dropout            = None,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None,
    lstm_kwargs        = None
):
    """Create version 3 LSTM network.

    This is the latest revision of the LSTM network:
        - It uses both 2D and 3D prong level inputs
        - It relies on a heavy input preprocessing and postprocessing.

    Parameters
    ----------
    lstm_units3d : int
        Number of units that LSTM layer that processes 3D prongs will have.
        Default: 16.
    lstm_units2d : int
        Number of units that LSTM layer that processes 2D prongs will have.
        Default: 16.
    layers_pre : list of int
        List of Dense layer sizes that will be used to preprocess prong inputs.
        Same Dense layer configuration will be used for 2D and 3D level
        prong inputs.
    layers_post : list of int
        List of Dense layer sizes that will be used to postprocess LSTM
        outputs.
    n_resblocks : int or None, optional
        Number of the fully connected residual blocks to be added before the
        output layer. Default: None
    max_prongs : int or None, optional
        Limit on the number of prongs that will be used. Default: None.
    reg : keras.Regularizer or None, optional
        Regularization to use. Default: None
    batchnorm : bool or None, optional
        Whether to use Batch Normalization. Default: False.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    vars_input_slice : list of str or None
        List of slice level input variable names.
    vars_input_png3d : list of str or None
        List of 3D prong level input variable names.
    vars_input_png2d : None
        List of 2D prong level input variable names.
        This is dummy input variable and MUST be None.
    var_target_total : str or None
        Name of the variable that defines true total energy.
    var_target_primary : str or None
        Name of the variable that defines true primary energy.
    lstm_kwargs : dict or None, optional
        Extra arguments that will be passed to the LSTM layer constructors.
        Default: None

    Returns
    -------
    keras.Model
        Model that defines the network.

    See Also
    --------
    model_lstm_v1
    model_lstm_v2
    model_lstm_v3_stack
    """

    # pylint: disable=dangerous-default-value

    inputs = get_inputs(
        vars_input_slice, vars_input_png3d, vars_input_png2d, max_prongs
    )
    # pylint: disable=unbalanced-tuple-unpacking
    input_slice, input_png3d, input_png2d = inputs

    layer_lstm_png3d = make_standard_lstm_branch(
        'png3d', input_png3d, layers_pre, lstm_units3d,
        batchnorm, dropout, reg, lstm_kwargs
    )

    layer_lstm_png2d = make_standard_lstm_branch(
        'png2d', input_png2d, layers_pre, lstm_units2d,
        batchnorm, dropout, reg, lstm_kwargs
    )

    layer_merged = Concatenate()([
        layer_lstm_png3d, layer_lstm_png2d, input_slice
    ])
    layer_merged = modify_layer(layer_merged, 'layer_merged', batchnorm)

    layer_post   = make_standard_postprocess_branch(
        layer_merged, layers_post, batchnorm, dropout, reg, n_resblocks
    )

    outputs = get_outputs(
        var_target_total, var_target_primary, reg, layer_post
    )

    return Model(inputs = inputs, outputs = outputs)

def model_lstm_v3_stack(
    lstm3d_spec        = [ (32, 'forward'), ],
    lstm2d_spec        = [ (32, 'forward'), ],
    layers_pre         = [],
    layers_post        = [],
    n_resblocks        = 0,
    max_prongs         = None,
    reg                = None,
    batchnorm          = False,
    dropout            = None,
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None,
    lstm_kwargs        = None
):
    """Create version 3 LSTM network that supports stacks of LSTM layers.

    Parameters
    ----------
    lstm3d_spec : list of (int, str)
        List of pairs that specify number of units and directions of LSTM
        layers that will be used to process 3D prongs.
        C.f. `add_stack_of_lstms`
    lstm2d_spec : list of (int, str)
        List of pairs that specify number of units and directions of LSTM
        layers that will be used to process 2D prongs.
        C.f. `add_stack_of_lstms`
    layers_pre : list of int
        List of Dense layer sizes that will be used to preprocess prong inputs.
        Same Dense layer configuration will be used for 2D and 3D level
        prong inputs.
    layers_post : list of int
        List of Dense layer sizes that will be used to postprocess LSTM
        outputs.
    n_resblocks : int or None, optional
        Number of the fully connected residual blocks to be added before the
        output layer. Default: None
    max_prongs : int or None, optional
        Limit on the number of prongs that will be used. Default: None.
    reg : keras.Regularizer or None, optional
        Regularization to use. Default: None
    batchnorm : bool or None, optional
        Whether to use Batch Normalization. Default: False.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    vars_input_slice : list of str or None
        List of slice level input variable names.
    vars_input_png3d : list of str or None
        List of 3D prong level input variable names.
    vars_input_png2d : None
        List of 2D prong level input variable names.
        This is dummy input variable and MUST be None.
    var_target_total : str or None
        Name of the variable that defines true total energy.
    var_target_primary : str or None
        Name of the variable that defines true primary energy.
    lstm_kwargs : dict or None, optional
        Extra arguments that will be passed to the LSTM layer constructors.
        Default: None

    Returns
    -------
    keras.Model
        Model that defines the network.

    See Also
    --------
    model_lstm_v3
    """
    # pylint: disable=dangerous-default-value

    inputs = get_inputs(
        vars_input_slice, vars_input_png3d, vars_input_png2d, max_prongs
    )
    # pylint: disable=unbalanced-tuple-unpacking
    input_slice, input_png3d, input_png2d = inputs

    layer_lstm_png3d = make_stacked_lstm_branch(
        'png3d', input_png3d, layers_pre, lstm3d_spec,
        batchnorm, dropout, reg, lstm_kwargs
    )

    layer_lstm_png2d = make_stacked_lstm_branch(
        'png2d', input_png2d, layers_pre, lstm2d_spec,
        batchnorm, dropout, reg, lstm_kwargs
    )

    layer_merged = Concatenate()([
        layer_lstm_png3d, layer_lstm_png2d, input_slice
    ])
    layer_merged = modify_layer(layer_merged, 'layer_merged', batchnorm)

    layer_post   = make_standard_postprocess_branch(
        layer_merged, layers_post, batchnorm, dropout, reg, n_resblocks
    )

    outputs = get_outputs(
        var_target_total, var_target_primary, reg, layer_post
    )

    return Model(inputs = inputs, outputs = outputs)


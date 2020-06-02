"""
Functions to calculate true and predicted energies.
"""

from lstm_ee.consts import ( LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY )
from lstm_ee.train.setup import get_keras_concurrency_kwargs

def _calc_secondary(result):
    """Calculate secondary energy"""
    if (
            (result[LABEL_TOTAL]   is not None)
        and (result[LABEL_PRIMARY] is not None)
    ):
        result[LABEL_SECONDARY] = result[LABEL_TOTAL] - result[LABEL_PRIMARY]
    else:
        result[LABEL_SECONDARY] = None

def predict_energies(args, dgen, model):
    """Calculate energies predicted by a `model`

    Parameters
    ----------
    args : Args
        Arguments that define `lstm_ee` training/evaluation.
    dgen : IDataGenerator
        Data generator which will be fed to the `model` to predict energies.
    model : `keras.Model`
        Model that will be used to predict energies.

    Returns
    -------
    dict
        Dictionary where labels are energy labels and values are `ndarray` of
        predicted energies.
        Possible labels are: [ LABEL_PRIMARY, LABEL_SECONDARY, LABEL_TOTAL ]
        If `model` is incapable of predicting certain energy type, then
        the corresponding values will be set to None.

    Notes
    -----
    Primary energy in the result will be calculated only if the `model` was
    trained to predict that energy. This is determined by looking at the
    `dgen.var_target_primary` and checking if it is not None.

    Total energy in the result will be calculated only if the `model` was
    trained to predict that energy. This is determined by looking at the
    `dgen.var_target_total` and checking if it is not None.

    Secondary energy is defined as (Total - Primary) and will be predicted
    only if the `model` predicts both primary and total energies.
    """
    kwargs = get_keras_concurrency_kwargs(args)
    pred   = model.predict_generator(dgen, **kwargs)

    result = {}
    idx    = 0

    if dgen.var_target_total is not None:
        total = pred[idx].ravel()
        idx += 1
    else:
        total = None

    if dgen.var_target_primary is not None:
        primary = pred[idx].ravel()
        idx += 1
    else:
        primary = None

    result[LABEL_TOTAL]   = total
    result[LABEL_PRIMARY] = primary

    _calc_secondary(result)

    return result

def get_true_energies(dgen):
    """Calculate true energies.

    Parameters
    ----------
    dgen : IDataGenerator
        Data generator which will be used to get true energies.

    Returns
    -------
    dict
        Dictionary where labels are energy labels and values are `ndarray` of
        the true energies.
        Possible labels are: [ LABEL_PRIMARY, LABEL_SECONDARY, LABEL_TOTAL ].

    Notes
    -----
    The true energies are fetches from `dgen.data_loader`. To determine the
    variable names of the true energies `dgen.var_target_primary` and
    `dgen.var_target_total` are used. If either of them is set to None,
    then the values of corresponding true energies will also be set to None.
    """

    result = {}

    if dgen.var_target_total is not None:
        total = dgen.data_loader.get(dgen.var_target_total).ravel()
    else:
        total = None

    if dgen.var_target_primary is not None:
        primary = dgen.data_loader.get(dgen.var_target_primary).ravel()
    else:
        primary = None

    result[LABEL_TOTAL]   = total
    result[LABEL_PRIMARY] = primary

    _calc_secondary(result)

    return result

def get_base_energies(dgen, pred_map):
    """Calculate baseline energies.

    Parameters
    ----------
    dgen : IDataGenerator
        Data generator which `dgen.data_loder` will be used to retrieve values
        of the baseline energies.
    pred_map : dict
        Dictionary that specifies mapping between energy label and a variable
        name in `dgen.data_loader` that holds baseline reconstructed energy.

    Returns
    -------
    dict
        Dictionary where labels are energy labels and values are `ndarray` of
        the baseline energies.
        Possible labels are: [ LABEL_PRIMARY, LABEL_SECONDARY, LABEL_TOTAL ].
    """

    if pred_map is None:
        return {
            k : None for k in [ LABEL_PRIMARY, LABEL_SECONDARY, LABEL_TOTAL ]
        }

    result = {
        k : dgen.data_loader.get(v).ravel() for k,v in pred_map.items()
    }
    _calc_secondary(result)

    return result


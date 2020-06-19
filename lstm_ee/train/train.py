"""
Functions to train `lstm_ee` model.
"""

import logging
import numpy as np

from lstm_ee.args import Args
from lstm_ee.data import load_data
from .setup       import (
    get_optimizer, get_default_callbacks, get_keras_concurrency_kwargs,
    select_model
)

LOGGER = logging.getLogger('lstm_ee.train')

def return_training_stats(train_log, savedir):
    """Return a dict with a summary of training results.

    Parameters
    ----------
    train_log : keras.History
        Training history returned by `keras.model.fit`
    savedir : str
        Directory where trained model is saved.

    Return
    ------
    dict
        Dictionary with training summary.
    """

    best_idx = np.argmin(train_log.history['val_loss'])

    ms   = train_log.history['val_target_total_cl_ms_relative_error'][best_idx]
    mean = \
        train_log.history['val_target_total_cl_mean_relative_error'][best_idx]

    stdev = np.sqrt(ms - mean**2)

    result = {
        'loss'    : (
            train_log.history['val_target_total_loss'][best_idx]
          + train_log.history['val_target_primary_loss'][best_idx]
        ),
        'rms'     : np.sqrt(ms),
        'mean'    : mean,
        'stdev'   : stdev,
        'status'  : 0,
        'time'    : train_log.history['train_time'][-1],
        'epochs'  : len(train_log.history['val_loss']),
        'savedir' : savedir
    }

    return result

def create_and_train_model(args = None, extra_kwargs = None, **kwargs):
    """Creates and trains `keras` model specified by arguments.

    Parameters
    ----------
    args : Args or None, optional
        Specification of the model and training setup
        If None, then the model and training specification will be first
        constructed from `kwargs` and `extra_kwargs`
    extra_kwargs : dict or None, optional
        Extra kwargs that will be passed to the `Args` constructor.
    kwargs : dict
        Parameters that will be passed to the `Args` constructor if `args` is
        None.

    Return
    ------
    dict
        Dictionary with training summary returned by `return_training_stats`.

    See Also
    --------
    lstm_ee.args.Args
    return_training_stats
    """

    if args is None:
        args = Args(extra_kwargs = extra_kwargs, **kwargs)

    LOGGER.info(
        "Starting training with parameters:\n%s", args.config.pprint()
    )

    LOGGER.info("Loading data...")
    dgen_train, dgen_test = load_data(args)

    LOGGER.info("Compiling model..")
    np.random.seed(args.seed)

    optimizer = get_optimizer(args.optimizer)
    model     = select_model(args)
    callbacks = get_default_callbacks(args)

    model.compile(
        loss      = args.config.loss,
        optimizer = optimizer,
        metrics   = [ 'mean_relative_error', 'ms_relative_error' ]
    )

    steps_per_epoch = None
    if args.steps_per_epoch is not None:
        steps_per_epoch = min(args.steps_per_epoch, len(dgen_train))

    LOGGER.info("Training model..")
    train_log = model.fit_generator(
        dgen_train,
        epochs          = args.epochs,
        steps_per_epoch = steps_per_epoch,
        validation_data = dgen_test,
        callbacks       = callbacks,
        **get_keras_concurrency_kwargs(args)
    )

    LOGGER.info("Training complete.")

    return return_training_stats(train_log, args.savedir)


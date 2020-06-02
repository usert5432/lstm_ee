"""
A collection of functions to setup keras training.
"""

import keras

from lstm_ee.keras.callbacks import TrainTime
from lstm_ee.keras.models    import (
    flattened_model, model_lstm_v1, model_lstm_v2, model_lstm_v3,
    model_slice_linear, model_lstm_v3_stack
)

def get_optimizer(optimizer):
    """Get `keras` optimizer from the conf stored in `optimizer`"""

    if not isinstance(optimizer, dict):
        return optimizer

    name   = optimizer['name']
    kwargs = optimizer.get('kwargs', {}) or {}

    if name.lower() == 'rmsprop':
        return keras.optimizers.RMSprop(**kwargs)

    elif name.lower() == 'adam':
        return keras.optimizers.Adam(**kwargs)

    else:
        raise ValueError("Unknown optimizer: %s" % (optimizer))

def get_schedule(schedule):
    """Get `keras` LR decay schedule from the conf stored in `schedule`"""

    if not isinstance(schedule, dict):
        return schedule

    name   = schedule['name']
    kwargs = schedule.get('kwargs', {}) or {}

    if name.lower() == 'standard':
        return keras.callbacks.ReduceLROnPlateau(**kwargs)

    elif name.lower() == 'custom':
        return keras.callbacks.LearningRateScheduler(**kwargs)

    else:
        raise ValueError("Unknown schedule: %s" % (schedule))

def get_early_stop(early_stop):
    """Get `keras` early stop from the conf stored in `early_stop`"""

    if not isinstance(early_stop, dict):
        return early_stop

    name   = early_stop['name']
    kwargs = early_stop.get('kwargs', {}) or {}

    if name.lower() == 'standard':
        return keras.callbacks.EarlyStopping(**kwargs)

    else:
        raise ValueError("Unknown early stoping: %s" % (early_stop))

def get_default_callbacks(args):
    """Get default `keras` callbacks for the `lstm_ee` training"""

    cb_checkpoint = keras.callbacks.ModelCheckpoint(
        "%s/model.h5" % args.savedir,
        monitor = 'val_loss',
        verbose = 0,
        save_best_only = args.save_best
    )

    cb_logger     = keras.callbacks.CSVLogger("%s/log.csv" % args.savedir)
    cb_time       = TrainTime()
    cb_schedule   = get_schedule(args.schedule)
    cb_early_stop = get_early_stop(args.early_stop)

    callbacks = [ cb_time, cb_checkpoint, cb_logger ]

    if cb_schedule is not None:
        callbacks.append(cb_schedule)

    if cb_early_stop is not None:
        callbacks.append(cb_early_stop)

    return callbacks

def get_regularizer(regularizer):
    """Get `keras` regularizer from the conf stored in `regularizer`"""

    if not isinstance(regularizer, dict):
        return regularizer

    name   = regularizer['name']
    kwargs = regularizer.get('kwargs', {}) or {}

    if name.lower() == 'l1':
        return keras.regularizers.l1(**kwargs)

    if name.lower() == 'l2':
        return keras.regularizers.l2(**kwargs)

    if name.lower() == 'l1_l2':
        return keras.regularizers.l1_l2(**kwargs)

    raise ValueError("Unknown regularizer: %s" % (regularizer))

def select_model(args):
    """Get `keras` model for the `lstm_ee` training."""

    name   = args.model['name']
    kwargs = args.model.get('kwargs', {}) or {}

    kwargs = {
        'reg'                 : get_regularizer(args.regularizer),
        'max_prongs'          : args.max_prongs,
        'vars_input_slice'    : args.vars_input_slice,
        'vars_input_png3d'    : args.vars_input_png3d,
        'vars_input_png2d'    : args.vars_input_png2d,
        'var_target_total'    : args.var_target_total,
        'var_target_primary'  : args.var_target_primary,
        **kwargs
    }

    if name == 'lstm_v1':
        return model_lstm_v1(**kwargs)
    if name == 'lstm_v2':
        return model_lstm_v2(**kwargs)
    if name == 'lstm_v3':
        return model_lstm_v3(**kwargs)
    if name == 'lstm_v3_stack':
        return model_lstm_v3_stack(**kwargs)
    if name == 'slice_linear':
        return model_slice_linear(**kwargs)
    if name == 'flattened':
        return flattened_model(**kwargs)
    else:
        raise ValueError("Unknown model name: %s" % (args.model))

def get_keras_concurrency_kwargs(args):
    """Get arg dict for setting up concurrency in `keras.model.fit`"""
    result = {}
    result['workers'] = 0

    if args.cache or (args.concurrency is None):
        return result

    if (args.workers is None) or (args.workers < 1):
        return result

    result['workers'] = args.workers

    if args.concurrency == 'process':
        result['use_multiprocessing'] = True
    elif args.concurrency == 'thread':
        result['use_multiprocessing'] = False
    else:
        raise RuntimeError(
            "Unknown parallelization type: %s" % (args.concurrency)
        )

    return result


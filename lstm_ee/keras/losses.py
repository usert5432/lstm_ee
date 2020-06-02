"""Custom `keras` losses"""

import keras.backend as K

def cl_mean_relative_error(y_true, y_pred):
    """Clipped Mean Relative Error"""
    diff = (
        (y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None)
    )

    diff = K.switch(K.less(K.abs(diff), 1), diff, K.zeros_like(diff))

    return K.mean(diff, axis=-1)

def cl_ms_relative_error(y_true, y_pred):
    """Clipped Mean Squared Relative Error"""
    diff = K.square(
        (y_true - y_pred)) / K.clip(K.square(y_true), K.epsilon(), None
    )

    diff = K.switch(K.less(diff, 1), diff, K.zeros_like(diff))

    return K.mean(diff, axis=-1)

def huber_loss(err, delta):
    """Huber function"""
    abserr = K.abs(err)
    res_lo = K.square(abserr) / 2
    res_hi = delta * (abserr - delta / 2)

    return K.switch(K.less(abserr, delta), res_lo, res_hi)

def relative_error_huber(y_true, y_pred, delta):
    """Huber relative error loss"""
    diff   = (y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None)
    result = huber_loss(diff, delta)

    return K.mean(result, axis=-1)

def error_huber(y_true, y_pred, delta):
    """Huber absolute error loss"""
    diff   = (y_true - y_pred)
    result = huber_loss(diff, delta)

    return K.mean(result, axis=-1)


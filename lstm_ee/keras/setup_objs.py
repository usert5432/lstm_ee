"""Functions to register custom `keras` objects to allow their serialization"""

from keras.utils.generic_utils import get_custom_objects
from .losses import (
    cl_mean_relative_error, cl_ms_relative_error,
    relative_error_huber, error_huber
)

def setup_keras_custom_objects():
    """Register custom `keras` losses"""
    get_custom_objects().update({
        "mean_relative_error"    : cl_mean_relative_error,
        "ms_relative_error"      : cl_ms_relative_error,
    })

    get_custom_objects().update({
        "relative_error_huber%.2f" % (delta) :
            lambda x, y, delta = delta : relative_error_huber(x, y, delta)
        for delta in [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5, 10
        ]
    })

    get_custom_objects().update({
        "error_huber%.2f" % (delta) :
            lambda x, y, delta = delta : error_huber(x, y, delta)
        for delta in [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5, 10
        ]
    })


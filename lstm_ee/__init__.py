"""
lstm_ee
=======

A package to assist with development of the LSTM energy estimators for NOvA.

Available subpackages
---------------------
args
    Contains structures to store persistent training configuration and various
    runtime options.
data
    Has multiple classes and functions to load datasets from a disk and make
    batches of data from them. It also defines various data transformations
    that might be useful.
eval
    Defines functions to assist with network evaluation.
keras
    This submodule contains definitions of the actual `keras` models, losses
    and callbacks.
plot
    A collection of routines to assist with plotting of evaluation results.
presets
    This submodule has predefined training/evaluation configurations that are
    commonly used.
train
    Has functions to help setup model training. And to perform the actual
    training.
utils
    A collection of helper utility functions.
"""

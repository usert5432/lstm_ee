"""
This module contains a collection of decorator patterns around DataGenerator.

These decorators are not `lstm_ee` specific and can be used with any
DataGenerator as long as it implements __len__ can __getitem__ function.

The __len__ function should return the number of batches that the DataGenerator
is able to generate. The __getitem__ function takes index of a batch and
returns a tuple of (inputs, targets) or (inputs, targets, weights) where
inputs and targets and dicts of { "label" : batch_values }.

The decorators in this module are indented to be used for both `lstm_ee` and
`slice_lid`.
"""


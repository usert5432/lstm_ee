"""
Data related functions and objects.

This module contains several parts:
    - `data_loader` defines a DataLoader objects that behave similar to pandas
      DataFrame and are used to load data. In addition it provides DataLoader
      wrappers that implement various transformations on the loaded dataset.
    - `data_generator` defined a DataGenerator object that takes DataLoader
      as an input and creating batches of data from it. Similar to
      `data_loader` this submodule also contains wrappers that apply
      transformation to generated batches of data.
    - `data` file defines a number of routines to simplify data handling.
"""


from .data import load_data, create_data_generators, construct_data_loader

__all__ = [ 'load_data', 'create_data_generators', 'construct_data_loader' ]


"""
Data related functions and objects.

This module contains several parts:
    - `data_loader` defines a DataLoader objects that behave similar to pandas
      DataFrame and are used to load data. In addition it provides DataLoader
      wrappers that implement various transformations on the loaded dataset.
    - `data_generator` defines a DataGenerator object that takes a DataLoader
      as input and creates batches of data from it. This submodule also
      defines a number of wrappers that apply transformation to the generated
      batches of data.
    - `data` file defines a number of routines to simplify data handling.
"""

from .data import load_data, create_data_generators, construct_data_loader

__all__ = [ 'load_data', 'create_data_generators', 'construct_data_loader' ]


"""
This module contains a number of objects for loading datasets from csv and hdf
files. It also contains dataset transformations.
"""

from .csv_loader   import CSVLoader
from .hdf_loader   import HDFLoader
from .dict_loader  import DictLoader
from .data_shuffle import DataShuffle
from .data_slice   import DataSlice

__all__ = [
    'CSVLoader', 'HDFLoader', 'DictLoader', 'DataShuffle', 'DataSlice'
]


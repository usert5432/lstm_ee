"""Convert custom CSV file into HDF file for `lstm_ee` training"""

import argparse
import warnings

import numpy as np
import tables

from lstm_ee.data.data_loader import CSVLoader

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser("Convert CSV to HDF")

    parser.add_argument(
        'input',
        help    = 'Input HDF file',
        metavar = 'input',
        type    = str,
        nargs   = '+'
    )

    parser.add_argument(
        '-o', '--output',
        help     = 'Output File',
        type     = str,
        required = True
    )

    return parser

class HDFExporter():
    """Object that saves `IDataLoader` variables into HDF file.

    Parameters
    ----------
    path : str
        Name of the HDF output file.
    """

    def __init__(self, path):
        self._path = path
        filters    = tables.Filters(complib = 'zlib', complevel = 5)
        self._f    = tables.open_file(path, 'w', filters = filters)

    @staticmethod
    def extend_hdf_dset(dset, size):
        """Extend existing HDF dataset for appending data to its end"""
        old_size = len(dset)
        dset.resize((old_size + size), axis = 0)

        return old_size

    def _export_scalar_var(self, var, data):
        """Save scalar data into HDF file"""
        node_name = '/' + var

        if node_name in self._f:
            node = self._f.get_node(node_name)
        else:
            node = self._f.create_earray(
                '/', var,
                atom  = tables.Atom.from_dtype(data.dtype),
                shape = (0,)
            )

        node.append(data)

    def _export_varr_var(self, var, data, dtype = np.float32):
        """Save variable length array data into HDF file"""
        # pylint: disable=unused-argument
        node_name = '/' + var

        if node_name in self._f:
            node = self._f.get_node(node_name)
        else:
            node = self._f.create_vlarray(
                '/', var, atom = tables.Float32Atom(shape=())
            )

        for idx,row in enumerate(data):
            if idx % 100 == 0:
                print(
                    "        %s : %d / %d" % (
                        var, idx, len(data)
                    ), end = '\r'
                )
            node.append(row)

    def export(self, loader):
        """Save all variables from `loader` into HDF file."""
        for var in loader.variables():
            print("        %s" % (var), end = '\r')
            data = loader.get(var)
            if np.issubdtype(data.dtype, np.number):
                self._export_scalar_var(var, data)
            else:
                self._export_varr_var(var, data)

    def close(self):
        """Close output HDF file"""
        self._f.close()

def main():
    # pylint: disable=missing-function-docstring
    warnings.filterwarnings('ignore', category = tables.NaturalNameWarning)

    parser  = create_parser()
    cmdargs = parser.parse_args()

    exporter = HDFExporter(cmdargs.output)

    for idx,path in enumerate(cmdargs.input):
        print("Processing file %d of %d" % (idx + 1, len(cmdargs.input)))
        print("   Loading...")
        loader = CSVLoader(path)
        print("   Exporting...")
        exporter.export(loader)

    exporter.close()
    print("Done")

if __name__ == '__main__':
    main()


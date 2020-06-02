"""Test correctness of custom csv files parsing with `CSVLoader`"""

import io
import unittest

from lstm_ee.data.data_loader.csv_loader import CSVLoader

from .tests_data_loader_base import TestsDataLoaderBase

def create_csv_data_str(data):
    """Create csv string representation of data from a `data` dict"""

    def export_value(value):
        if isinstance(value, list):
            return '"%s"' % (",".join([str(x) for x in value]))

        return str(value)

    columns = list(data.keys())
    length  = len(data[columns[0]])

    result = io.StringIO()
    result.write(",".join(columns) + '\n')

    for i in range(length):
        result.write(
            ",".join([ export_value(data[c][i]) for c in columns ]) + '\n'
        )

    result.seek(0)

    return result

class TestsCSVLoader(TestsDataLoaderBase, unittest.TestCase):
    """Test `CSVLoader` data parsing"""

    def _create_data_loader(self, data):
        csv_data = create_csv_data_str(data)
        return CSVLoader(csv_data)

if __name__ == '__main__':
    unittest.main()


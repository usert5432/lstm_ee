"""
Test runner to run all available tests.

Examples
--------
Run all available tests:

$ python -m unittest tests.run_tests.suite
"""

import unittest

import tests.data_loader.tests_csv_loader
import tests.data_loader.tests_hdf_loader
import tests.data_loader.tests_dict_loader
import tests.data_loader.tests_data_shuffle
import tests.data_loader.tests_data_slice

import tests.data_generator.tests_batch_split
import tests.data_generator.tests_varr_sorting
import tests.data_generator.tests_noise
import tests.data_generator.tests_weights

def suite():
    """Create test suite"""
    result  = unittest.TestSuite()
    loader = unittest.TestLoader()

    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_csv_loader
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_hdf_loader
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_dict_loader
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_data_shuffle
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_data_slice
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_batch_split
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_varr_sorting
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_noise
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_weights
    ))

    return result

def run():
    """Run test suite"""
    runner = unittest.TextTestRunner(verbosity = 3)
    runner.run(suite())

if __name__ == '__main__':
    run()


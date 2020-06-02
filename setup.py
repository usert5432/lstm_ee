#!/usr/bin/env python

import setuptools

def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(
    name             = 'lstm_ee',
    version          = '0.1.0',
    author           = 'Dmitrii Torbunov',
    author_email     = 'torbu001@umn.edu',
    classifiers      = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = 'Package to train NOvA LSTM Energy Estimators',
    install_requires = [
        'cafplot',
        'keras',
        'pandas',
        'scipy',
        'cython',
    ],
    license          = 'MIT',
    long_description = readme(),
    packages         = setuptools.find_packages(),
    url              = 'https://github.com/usert5432/lstm_ee',
)


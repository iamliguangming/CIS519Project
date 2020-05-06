from distutils.core import setup
from os.path import isdir
from itertools import product

packages = ['flightsim','generator']

setup(
    name='aerial_robotics',
    packages=packages,
    version='0.1',
    install_requires=[
            'cvxopt',
            'matplotlib',
            'numpy',
            'scipy',
            'timeout_decorator'])

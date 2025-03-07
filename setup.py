#!/usr/bin/env python
from setuptools import setup

setup(
    name='emulator-configurable',
    version='0.0.1',
    python_requires='>3.6',
    packages=['emulator_configurable'],
    install_requires=[
        'torch',
        'pytorch-lightning',
        'numpy',
        'xarray',
        'pandas',
        'dask[diagnostics]',
        'bottleneck',
        'mlflow',
        'xbatcher',
    ],
    entry_points={'console_scripts': [
        'parflow_emulator= emulator_configurable.main:main',
    ]},
)

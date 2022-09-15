#!/usr/bin/env python
from setuptools import setup

setup(
    name='emulator-configurable',
    version='0.0.0',
    python_requires='>3.6',
    packages=['emulator'],
    install_requires=[
        'torch',
        'pytorch-lightning',
        'numpy',
        'xarray',
        'pandas',
        'dask[diagnostics]',
        'bottleneck',
        'mlflow',
    ],
    entry_points={'console_scripts': [
        'run_emulator= emulator.cli:main',
    ]},
)

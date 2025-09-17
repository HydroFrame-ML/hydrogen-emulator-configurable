import emulator_configurable as emulator
import argparse
import json
import os
import sys
import torch
import shutil
import xarray as xr
from functools import partial
from pprint import pprint

from . import scalers
from .utils import (
    maybe_split_3d_vars,
    try_get_checkpoint,
    save_predictions
)

# Set to suppress a warning, doesn't seem to have a performance impact
torch.set_float32_matmul_precision('medium')# | 'high')


def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,
        help='Mode to run (either train or predict)')
    parser.add_argument('-c', '--config', type=str,
        help='Path to a configuration file')
    return parser.parse_args(args)


def main():
    args = parse(sys.argv[1:])
    mode = args.mode

    assert mode in ['train', 'predict'], (
            'Mode must be either train or predict!')
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        config['config_file'] = args.config
        if "selectors" in config:
            config['selectors'] = {
                k: slice(v.get('start', 0), v.get('stop', None), v.get('step', None)) 
                for k, v in config['selectors'].items()
            }
    if mode == 'train':
        emulator.train.train_model(**config)
    elif mode == 'predict':
        raise NotImplementedError()

if __name__ == '__main__':
    import warnings
    import logging
    warnings.filterwarnings('ignore', message='.*garbage collection.*')
    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.ERROR)
    main()

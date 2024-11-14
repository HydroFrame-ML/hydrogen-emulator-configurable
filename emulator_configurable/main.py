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

def predict_surface(
    config: dict,
):
    raise NotImplementedError()


def predict_subsurface(
    config: dict,
):
    """
    Run subsurface prediction using the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for subsurface prediction.

    Returns:
        pred_ds (xarray.Dataset): The predicted subsurface dataset.
    """
    # Get the model weights
    checkpoint_location = config.get('logging_location', 'https://concord.princeton.edu/mlflow/')
    model_weights_file = try_get_checkpoint(
        config['run_name'],
        checkpoint_location, 
        checkpoint_dir=config.get('checkpoint_dir', '.')
    )
    config['model_weights'] = torch.load(model_weights_file)['state_dict']

    # Open the data, do some preprocessing
    selectors = config.get('selectors', {})
    ds = xr.open_mfdataset(config['inference_dataset_files'], engine='zarr').bfill('time')
    ds = maybe_split_3d_vars(ds).isel(**selectors)

    # Run inference
    pred_ds = emulator.inference.run_subsurface_inference(ds, **config)

    # Save the results
    if 'save_path' in config:
        save_predictions(pred_ds, config['save_path'])
    return pred_ds


def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,
        help='Mode to run (either train or predict)')
    parser.add_argument('-d', '--domain', type=str,
        help='Domain to run (one of surface, subsurface, or combined)')
    parser.add_argument('-c', '--config', type=str,
        help='Path to a configuration file')
    return parser.parse_args(args)


def main():
    args = parse(sys.argv[1:])
    mode = args.mode
    domain = args.domain
    assert mode in ['train', 'predict'], (
            'Mode must be either train or predict!')
    assert domain in ['surface', 'subsurface'], (
            'Domain must be "surface", "subsurface", or "combined"!')
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        config['config_file'] = args.config
    if mode == 'train' and domain == 'surface':
        emulator.train.train_model(**config)
    elif mode == 'train' and domain == 'subsurface':
        emulator.train.train_model(**config)
    elif mode == 'predict' and domain == 'surface':
        predict_surface(**config)
    elif mode == 'predict' and domain == 'subsurface':
        predict_subsurface(config)

if __name__ == '__main__':
    main()

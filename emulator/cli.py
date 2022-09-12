import emulator
import argparse
import json
import os
import sys
import mlflow
from functools import partial
from pprint import pprint
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)
from hydroml.utils import MetricsCallback
from hydroml import scalers

def train_surface(
    config: dict,
):
    mlflow.set_tracking_uri(f'file:{config["log_dir"]}')
    base_data_gen = emulator.utils.zarr_data_gen
    config['train_data_gen_function'] = partial(
        base_data_gen,  selectors={'time': slice(0, -300)}
    )
    config['valid_data_gen_function'] = partial(
        base_data_gen,  selectors={'time': slice(-300, None)}
    )
    lr_monitor = LearningRateMonitor(
            logging_interval='step',
    )
    metrics = MetricsCallback()
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        every_n_train_steps=config['logging_frequency'],
        every_n_epochs=None,
        monitor='train_loss'
    )
    config['callbacks'] = [metrics, checkpoint, lr_monitor]
    scaler_file = config.pop('scaler_file')
    config['scalers'] = scalers.load_scalers(scaler_file)
    emulator.train.train_surface_model(config)



def train_subsurface(
    config: dict,
):
    mlflow.set_tracking_uri(f'file:{config["log_dir"]}')
    base_data_gen = emulator.utils.zarr_data_gen
    config['train_data_gen_function'] = partial(
        base_data_gen, selectors={'time': slice(0, -300)}
    )
    config['valid_data_gen_function'] = partial(
        base_data_gen, selectors={'time': slice(-300, None)}
    )
    lr_monitor = LearningRateMonitor(
            logging_interval='step',
    )
    metrics = MetricsCallback()
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        every_n_train_steps=config['logging_frequency'],
        every_n_epochs=None,
        monitor='train_loss'
    )
    config['callbacks'] = [metrics, checkpoint, lr_monitor]
    scaler_file = config.pop('scaler_file')
    config['scalers'] = scalers.load_scalers(scaler_file)
    config['scalers']['pressure_1'] = config['scalers']['pressure']
    emulator.train.train_model(config)


def train_single_layer(
    config: dict,
):
    raise NotImplementedError()

def train_combined(
    config: dict,
):
    raise NotImplementedError()


def predict_surface(
    config: dict,
):
    raise NotImplementedError()


def predict_subsurface(
    config: dict,
):
    raise NotImplementedError()


def predict_combined(
    config: dict,
):
    raise NotImplementedError()


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
    assert domain in ['surface', 'subsurface', 'layer', 'combined'], (
            'Domain must be "surface", "subsurface", or "combined"!')
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
    if mode == 'train' and domain == 'surface':
        train_surface(config)
    elif mode == 'train' and domain == 'subsurface':
        train_subsurface(config)
    elif mode == 'train' and domain == 'layer':
        train_single_layer(config)
    elif mode == 'train' and domain == 'combined':
        train_combined(config)
    elif mode == 'predict' and domain == 'surface':
        predict_surface(config)
    elif mode == 'predict' and domain == 'subsurface':
        predict_subsurface(config)
    elif mode == 'predict' and domain == 'layer':
        raise NotImplementedError()
    elif mode == 'predict' and domain == 'combined':
        predict_combined(config)


if __name__ == '__main__':
    main()

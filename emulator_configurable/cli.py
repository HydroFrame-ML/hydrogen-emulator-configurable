import emulator_configurable as emulator
import argparse
import json
import os
import sys
import mlflow
import shutil
import hydroml as hml
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
    config['train_files'] = [
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2003_preprocessed.zarr',
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2004_preprocessed.zarr',
    ]
    config['valid_files'] = [
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2005_preprocessed.zarr',
    ]
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
    emulator.train.train_model(config)


def train_subsurface(
    config: dict,
):
    mlflow.set_tracking_uri(f'file:{config["log_dir"]}')
    config['train_files'] = [
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2003_preprocessed.zarr',
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2004_preprocessed.zarr',
    ]
    config['valid_files'] = [
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2005_preprocessed.zarr',
    ]
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
    import dask
    from dask.distributed import Client, LocalCluster
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    cluster = LocalCluster(
        n_workers=12,
        threads_per_worker=2,
        memory_limit='96GB',
        diagnostics_port=':3878'
    )
    client = Client(cluster)
    print(client)
    base_data_gen = emulator.utils.zarr_data_gen
    in_files = [
        '/scratch/ab6361/pfclm_conus1_zarr/conus1_2006_preprocessed.zarr',
    ]
    ds = base_data_gen(files=in_files).chunk(dict(time=1, x=512, y=512))
    #scaler_file = config.pop('scaler_file')
    #config['scalers'] = scalers.load_scalers(scaler_file)
    pred_ds = emulator.forecast.run_subsurface_forecast(ds, config)
    if 'save_path' in config and os.path.exists(config['save_path']):
        shutil.rmtree(config['save_path'])
    if 'save_path' in config and config['save_path'].endswith('zarr'):
        pred_ds.to_zarr(config['save_path'], consolidated=True)
    elif 'save_path' in config and config['save_path'].endswith('nc'):
        pred_ds.to_netcdf(config['save_path'])
    return pred_ds



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

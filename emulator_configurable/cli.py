import emulator_configurable as emulator
import argparse
import json
import os
import sys
import shutil
from functools import partial
from pprint import pprint

from . import scalers

def predict_surface(
    config: dict,
):
    raise NotImplementedError()


def predict_subsurface(
    config: dict,
):
    """
    TODO: go back and refactor this to be more flexible on the data input side
    """
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
    pred_ds = emulator.forecast.run_subsurface_forecast(ds, config)
    if 'save_path' in config and os.path.exists(config['save_path']):
        shutil.rmtree(config['save_path'])
    if 'save_path' in config and config['save_path'].endswith('zarr'):
        pred_ds.to_zarr(config['save_path'], consolidated=True)
    elif 'save_path' in config and config['save_path'].endswith('nc'):
        pred_ds.to_netcdf(config['save_path'])
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
        predict_surface(config)
    elif mode == 'predict' and domain == 'subsurface':
        predict_subsurface(config)

if __name__ == '__main__':
    main()

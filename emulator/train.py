import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import hydroml as hml
from torch import nn
import pytorch_lightning as pl
from glob import glob
from .dataset import RecurrentDataset, worker_init_fn
from .model_builder import ModelBuilder
from hydroml.process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)
from torch.utils.data import DataLoader

def train_model(
    config,
    callbacks=None
):
    resume_from_checkpoint = config['resume_from_checkpoint']
    log_dir = config['log_dir']
    run_name = config['run_name']
    logging_frequency = config['logging_frequency']

    patch_size = config['patch_size']
    patch_stride = config['patch_stride']
    scalers = config['scalers']
    train_data_gen_function = config['train_data_gen_function']
    valid_data_gen_function = config['valid_data_gen_function']
    forcing_vars = config['forcing_vars']
    state_vars = config['state_vars']
    surface_parameters = config['surface_parameters']
    subsurface_parameters = config['subsurface_parameters']
    out_vars = config['out_vars']

    sequence_length = config['sequence_length']
    train_samples_per_epoch = config['train_samples_per_epoch']
    valid_samples_per_epoch = config['valid_samples_per_epoch']
    batch_size = config['batch_size']
    num_dl_workers = config['num_dl_workers']
    max_epochs = config['max_epochs']
    run_name = config['run_name']
    log_dir = config['log_dir']


    patch_sizes = {'x': patch_size, 'y': patch_size}
    patch_strides = {'x': patch_stride, 'y': patch_stride}
    conus_scalers = scalers

    # Create the dataset object
    train_dataset = RecurrentDataset(
        data_gen_function=train_data_gen_function,
        static_inputs=surface_parameters+subsurface_parameters,
        forcing_inputs=forcing_vars,
        state_inputs=state_vars,
        dynamic_outputs=out_vars,
        scalers=conus_scalers,
        sequence_length=sequence_length,
        patch_sizes=patch_sizes,
        patch_strides=patch_strides,
        samples_per_epoch_total=train_samples_per_epoch,
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_dl_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    valid_dataset = RecurrentDataset(
        data_gen_function=valid_data_gen_function,
        static_inputs=surface_parameters+subsurface_parameters,
        forcing_inputs=forcing_vars,
        state_inputs=state_vars,
        dynamic_outputs=out_vars,
        scalers=conus_scalers,
        sequence_length=sequence_length,
        patch_sizes=patch_sizes,
        patch_strides=patch_strides,
        samples_per_epoch_total=valid_samples_per_epoch
    )
    valid_dataset.per_worker_init()
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_dl_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    model_config = config.get('model_def', {}).get('model_config', {})
    model_config['forcing_vars'] = config['forcing_vars']
    model_config['surface_parameters'] = config['surface_parameters']
    model_config['subsurface_parameters'] = config['subsurface_parameters']
    model_config['state_vars'] = config['state_vars']
    model_config['out_vars'] = config['out_vars']
    model_config['sequence_length'] = config['sequence_length']
    from pprint import pprint

    # TODO: FIXME: Replace with model builder code
    model = ModelBuilder.build_emulator(
        emulator_type=config['model_def']['type'],
        model_config=config['model_def']['model_config']
    )
    logger = pl.loggers.MLFlowLogger(
        experiment_name=run_name,
        tracking_uri=f'file:{log_dir}',
    )
    checkpoint_dir = (f'{logger.save_dir}/{logger.experiment_id}'
                      f'/{logger.run_id}/checkpoints')
    hparams = {
        'checkpoint_dir': checkpoint_dir,
        'forcing_vars': forcing_vars,
        'surface_parameters': surface_parameters,
        'subsurface_parameters': subsurface_parameters,
        'state_vars': state_vars,
        'out_vars': out_vars,
        'patch_sizes': patch_sizes,
        'patch_strides': patch_strides,
        'sequence_length': sequence_length,
        'logger_frequency': logging_frequency,
        'train_samples_per_epoch': train_samples_per_epoch,
        'valid_samples_per_epoch': valid_samples_per_epoch,
    }
    logger.log_hyperparams(hparams)

    if (os.path.exists(str(resume_from_checkpoint))
        and os.path.isfile(str(resume_from_checkpoint))):
        print(f'Loading state dict from: {resume_from_checkpoint}')
        state_dict = torch.load(resume_from_checkpoint)
        model.load_state_dict(state_dict)
        ckpt_path = None
    elif resume_from_checkpoint:
        try:
            ckpt_path = hml.utils.find_resume_checkpoint(log_dir, run_name)
        except:
            print('-------------------------------------------------------')
            print(f'Could not find checkpoint!!!')
            print('-------------------------------------------------------')
            ckpt_path = None
    else:
        ckpt_path = None
    model.configure_loss()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        callbacks=callbacks,
        precision=32,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=logging_frequency,
        logger=logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
        ckpt_path=ckpt_path
    )

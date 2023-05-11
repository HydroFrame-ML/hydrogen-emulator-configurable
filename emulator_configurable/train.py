import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import hydroml as hml
from torch import nn
import pytorch_lightning as pl
from glob import glob
from .datapipes import create_new_loader
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
):
    resume_from_checkpoint = config['resume_from_checkpoint']
    log_dir = config['log_dir']
    run_name = config['run_name']

    logging_frequency = config.get('logging_frequency', 1)
    callbacks = config.get('callbacks', [])
    max_epochs = config.get('max_epochs', 1)

    # Create the dataset object
    train_dl = create_new_loader(
        files=config['train_dataset_files'],
        scaler_file=config['scaler_file'],
        nt=config['sequence_length'],
        ny=config['patch_size'],
        nx=config['patch_size'],
        forcings=config['forcings'],
        parameters=config['parameters'],
        states=config['states'],
        targets=config['targets'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        selectors={}, # {'x': slice(0, 640), 'y': slice(0, 640)} #TODO: remove this
    )

    # Create the dataset object
    valid_dl = create_new_loader(
        files=config['valid_dataset_files'],
        scaler_file=config['scaler_file'],
        nt=config['sequence_length'],
        ny=config['patch_size'],
        nx=config['patch_size'],
        forcings=config['forcings'],
        parameters=config['parameters'],
        states=config['states'],
        targets=config['targets'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        #selectors={}, # {'x': slice(0, 640), 'y': slice(0, 640)} #TODO: remove this
        selectors={}
    )

    model = ModelBuilder.build_emulator(
        type=config['model_def']['type'],
        config=config['model_def']['config']
    )
    logger = pl.loggers.MLFlowLogger(
        experiment_name=run_name,
        tracking_uri=f'file:{log_dir}',
    )
    checkpoint_dir = (f'{logger.save_dir}/{logger.experiment_id}'
                      f'/{logger.run_id}/checkpoints')
    print(log_dir, run_name)
    hparams = {
        'checkpoint_dir': checkpoint_dir,
        'forcings': config['forcings'],
        'parameters': config['parameters'],
        'states': config['states'],
        'targets': config['targets'],
        'patch_size': config['patch_size'],
        'sequence_length': config['sequence_length'],
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
            print('-------------------------------------------------------')
            print(f'Loading state dict from: {ckpt_path}')
            print('-------------------------------------------------------')
        except:
            raise
            print('-------------------------------------------------------')
            print(f'Could not find checkpoint for {run_name}!!!')
            print('-------------------------------------------------------')
            ckpt_path = None
    else:
        ckpt_path = None
    model.configure_loss()

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        precision=32,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=logging_frequency,
        logger=logger,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value"
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
        ckpt_path=ckpt_path
    )

import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F


from glob import glob
from . import utils
from .datapipes import create_new_loader
from .model_builder import ModelBuilder, model_setup
from .process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)
from .utils import MetricsCallback, find_resume_checkpoint

def train_model(
    config,
):
    # Set up logging
    logger = pl.loggers.MLFlowLogger(
        experiment_name=config['run_name'],
        tracking_uri="https://concord.princeton.edu/mlflow/"
    )
    logger.log_hyperparams(config)

    # Set up callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    metrics = MetricsCallback()
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        every_n_train_steps=config['logging_frequency'],
        every_n_epochs=None,
        monitor='train_loss'
    )
    epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=1,
    )
    callbacks = [lr_monitor, metrics, checkpoint, epoch_checkpoint]

    # Pull out some useful variables from the config
    resume_from_checkpoint = config['resume_from_checkpoint']
    log_dir = config['log_dir']
    run_name = config['run_name']
    # Might need to populate some defaults if not in the config
    logging_frequency = config.get('logging_frequency', 1)
    callbacks = config.get('callbacks', [])
    max_epochs = config.get('max_epochs', 1)
    device = config.get('device', 'gpu')

    if resume_from_checkpoint:
        ckpt_path = find_resume_checkpoint(run_name)
    else:
        ckpt_path = None


    # Set up the model 
    model = model_setup(
        model_type=config['model_def']['type'],
        model_config=config['model_def']['config'],
        learning_rate=config['learning_rate'],
        gradient_loss_penalty=config['gradient_loss_penalty'],
    )

    # Create the data loading pipeline
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
        selectors={},
    )

    # Configure the trainer. 
    trainer = pl.Trainer(
        accelerator=device,
        callbacks=callbacks,
        precision=config['precision'],
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=logging_frequency,
        logger=logger,
        gradient_clip_val=1.5,
        gradient_clip_algorithm="norm"
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        ckpt_path=ckpt_path
    )

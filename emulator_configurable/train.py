import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import utils

from glob import glob
from .datapipes import create_new_loader
from .model_builder import ModelBuilder
from .process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)

def train_model(
    config,
):
    # Set up logging
    logger = pl.loggers.MLFlowLogger(
        experiment_name=run_name,
        tracking_uri="http://concord.princeton.edu:5001"
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

   
    # Pull out some useful variables from the config
    resume_from_checkpoint = config['resume_from_checkpoint']
    log_dir = config['log_dir']
    run_name = config['run_name']
    # Might need to populate some defaults if not in the config
    logging_frequency = config.get('logging_frequency', 1)
    callbacks = config.get('callbacks', [])
    max_epochs = config.get('max_epochs', 1)
    device = config.get('device', 'gpu')

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

    # Create the model
    model = ModelBuilder.build_emulator(
        type=config['model_def']['type'],
        config=config['model_def']['config']
    )
    model.learning_rate = config['learning_rate']
    model.number_batches = train_dl.number_batches

    # Try to load a checkpoint from the provided path. If the path is a file,
    # load the state dict from the file. If just given a True, then 
    # try to find the last checkpoint in the log directory.
    if (os.path.exists(str(resume_from_checkpoint))
        and os.path.isfile(str(resume_from_checkpoint))):
        print(f'Loading state dict from: {resume_from_checkpoint}')
        state_dict = torch.load(resume_from_checkpoint)
        model.load_state_dict(state_dict)
        ckpt_path = None
    elif resume_from_checkpoint:
        try:
            ckpt_path = utils.find_last_checkpoint(log_dir, run_name)
            print('-------------------------------------------------------')
            print(f'Loading state dict from: {ckpt_path}')
            print('-------------------------------------------------------')
        except:
            print('-------------------------------------------------------')
            print(f'Could not find checkpoint for {run_name}!!!')
            print('-------------------------------------------------------')
            ckpt_path = None
    else:
        ckpt_path = None

    # Configure the loss function. If gradient_loss_penalty is True, use the
    # spatial gradient penalty loss, otherwise use the default mse_loss.
    # The spatial gradient penalty loss adds an additional term to the 
    # loss which accounts for the spatial gradient of the output, calculated
    # via a simple finite difference.
    if config['gradient_loss_penalty']:
        model.configure_loss(loss_fun=utils.spatial_gradient_penalty_loss)
    else:
        model.configure_loss(loss_fun=F.mse_loss)

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

import dask
import torch
import logging
import pytorch_lightning as pl

from dask.distributed import Client, LocalCluster
from typing import List, Union, Optional
from torch.utils.data import DataLoader
from .dataset import HydrogenDataset, BatchedDataset
from .model_builder import model_setup
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateMonitor
)
from .utils import (
    MetricsCallback,
    get_checkpoint_from_database,
    get_checkpoint_from_local_logs
)

dask.config.set(scheduler='synchronous')

def train_model(
    run_name: str,
    model_type: str,
    model_config: dict,
    forcings: List[str],
    parameters: List[str],
    states: List[str],
    targets: List[str],
    train_dataset_files: List[str],
    patch_size: int,
    max_epochs: int,
    learning_rate: float,
    sequence_length: int,
    *,
    selectors: dict={},
    batch_size: int=1,
    num_workers: int=1,
    precision: str='16',
    resume_from_checkpoint: Union[bool, str]=False,
    gradient_loss_penalty: bool=True,
    logging_frequency: int=10,
    callbacks: List[Callback]=[],
    device: Union[torch.device, str]='cuda',
    logging_location: str='https://concord.princeton.edu/mlflow/',
    scaler_file: Optional[str]=None,
    config_file: Optional[str]=None
):
    # Set up the cluster
    cluster = LocalCluster(
        n_workers=num_workers, threads_per_worker=1, memory_limit='4GB',
        dashboard_address=':4321',
    )
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")

    # Set up callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    metrics = MetricsCallback()
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        every_n_train_steps=logging_frequency,
        every_n_epochs=None,
        monitor='train_loss'
    )
    callbacks = [lr_monitor, metrics, checkpoint]

    # Get the checkpoint if we're resuming a training run
    if resume_from_checkpoint and isinstance(resume_from_checkpoint, bool):
        ckpt_path = get_checkpoint_from_database(
            run_name,
            logging_location
        )
    elif resume_from_checkpoint and isinstance(resume_from_checkpoint, str):
        ckpt_path = get_checkpoint_from_local_logs(
            run_name,
            logging_location,
            resume_from_checkpoint
        )
    else:
        ckpt_path = None

    # Set up logging
    logger = pl.loggers.MLFlowLogger(
        experiment_name=run_name,
        tracking_uri=logging_location,
        log_model=True
    )
    logger.log_hyperparams(locals())
    if config_file:
        logger.experiment.log_artifact(logger.run_id, config_file, )

    # Set up the model 
    model = model_setup(
        model_type=model_type,
        model_config=model_config,
        learning_rate=learning_rate,
        gradient_loss_penalty=gradient_loss_penalty,
    ).to(device)

    # Create the data loading pipeline
    ds = HydrogenDataset(
        train_dataset_files, sequence_length, patch_size, patch_size, 
        forcings, parameters, states, targets,
        selectors=selectors,
    )
    data_loader = BatchedDataset(
        ds, batch_size=batch_size, client=client, prefetch_factor=8
    )

    # Configure the trainer. 
    trainer = pl.Trainer(
        accelerator=device,
        devices=[0],
        callbacks=callbacks,
        precision=precision,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=logging_frequency,
        logger=logger,
        gradient_clip_val=1.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=8
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=data_loader,
        ckpt_path=ckpt_path
    )

    logger.finalize()
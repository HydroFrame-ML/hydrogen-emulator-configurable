import dask
import torch
import torch.nn.functional as F

import os
import json
import torch
import tempfile
import mlflow
from glob import glob
from tqdm.autonotebook import tqdm
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar


dask.config.set(**{'array.slicing.split_large_chunks': True})

def get_dtype(dtype):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Data type {dtype} not supported")

def get_optimizer(optimizer_type, model, learning_rate, **kwargs):
    if optimizer_type == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.025, **kwargs
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, **kwargs
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")
    return optimizer

def get_loss(loss_type):
    if loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "mae":
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f"Loss {loss_type} not supported")
    return loss_fn



# WARNING: All stuff below here from original hydrogen emulator

def update_config_for_inference(
        config,
        inference_dataset_files,
        save_path,
        selectors
):
    # Update the config to reflect the inference settings
    config['inference_dataset_files'] = inference_dataset_files
    config['save_path'] = save_path
    config['selectors'] = selectors
    config['states'] = config.pop('targets')

    stuff_we_dont_need_for_inference = [
        'sequence_length',
        'precision',
        'logging_frequency',
        'learning_rate',
        'batch_size',
        'num_workers',
        'num_epochs',
        'gradient_loss_penalty',
        'patch_size'
    ]
    for key in stuff_we_dont_need_for_inference:
        config.pop(key, None)
    return config

def save_predictions(ds, save_path):
    if save_path.endswith('zarr'):
        ds.to_zarr(save_path, consolidated=True)
    elif save_path.endswith('nc'):
        ds.to_netcdf(save_path)

def maybe_split_3d_vars(ds):
    """
    Splits 3D variables in the given dataset along the 'z' dimension.

    This function iterates over the variables in the dataset and checks if any of them have a dimension named 'z'.
    If a variable has a 'z' dimension, it splits the variable into multiple variables along the 'z' dimension.
    The new variables are named by appending the index of the 'z' dimension to the original variable name.

    Parameters:
    - ds (xarray.Dataset): The dataset containing the variables to be split.

    Returns:
    - ds (xarray.Dataset): The dataset with the 3D variables split along the 'z' dimension.

    Example:
    >>> ds = xr.Dataset({'pressure': (['z', 'y', 'x'], np.random.rand(10, 10, 5))})
    >>> ds = maybe_split_3d_vars(ds)
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:        (x: 10, y: 10, z: 5)
    Coordinates:
      * z              (z) int64 0 1 2 3 4
      * y              (y) int64 0 1 2 3 4 5 6 7 8 9
      * x              (x) int64 0 1 2 3 4 5 6 7 8 9
    Data variables:
        pressure   (z, y, x) float64 ...
        pressure_0    (x, y) float64 ...
        pressure_1    (x, y) float64 ...
        pressure_2    (x, y) float64 ...
        pressure_3    (x, y) float64 ...
        pressure_4    (x, y) float64 ...
    """
    for v in set(ds.variables) - set(ds.coords):
        if 'z' in ds[v].dims:
            for i in range(ds.sizes['z']):
                if f'{v}_{i}' not in ds:
                    ds[f'{v}_{i}'] = ds[v].isel(z=i)
    return ds

def spatial_gradient_penalty_loss(yhat, ytru, space_weight=1, loss_fun=F.mse_loss):
    """
    Calculates the spatial gradient penalty loss between predicted and true values.

    Args:
        yhat (torch.Tensor): The predicted values.
        ytru (torch.Tensor): The true values.
        space_weight (float, optional): The weight for the spatial gradient penalty. Defaults to 1.
        loss_fun (function, optional): The loss function to calculate the loss. Defaults to F.mse_loss.

    Returns:
        torch.Tensor: The calculated loss.
    """
    loss = loss_fun(ytru, yhat)

    dx_tru = torch.diff(ytru, dim=-1)
    dx_hat = torch.diff(yhat, dim=-1)
    dx_loss = loss_fun(dx_tru, dx_hat)

    dy_tru = torch.diff(ytru, dim=-2)
    dy_hat = torch.diff(yhat, dim=-2)
    dy_loss = loss_fun(dy_tru, dy_hat)

    return loss + space_weight * (dx_loss + dy_loss)


def count_parameters(model):
    """
    Returns the number of parameters in a pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_to_device(seq, device):
    """
    Move a sequence of tensors to a device.
    """
    return [s.to(device) for s in seq]


def match_dims(x, target):
    """
    Reshape x to match the dimensions of target.
    """
    return x.reshape([len(x) if i == len(x) else 1 for i in target.shape])


class MetricsCallback(Callback):
    """
    PyTorch Lightning metric callback.
    """

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def train_epoch_end(self, trainer, pl_module):
        for k, v in trainer.logged_metrics.items():
            if k not in self.metrics.keys():
                self.metrics[k] = [self._convert(v)]
            else:
                self.metrics[k].append(self._convert(v))

    def _convert(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        return x


class LitProgressBar(TQDMProgressBar):
    """
    This just avoids a bug in the progress bar for pytorch lightning
    that causes the progress bar to creep down the notebook
    """
    def init_validation_tqdm(self):
        bar = tqdm(disable=True,)
        return bar


# MLFLOW Utils:
def load_mlflow_credentials(mlflow_credentials_file):
    """
    Loads MLflow credentials from a file and sets them as environment variables.

    Args:
        mlflow_credentials_file (str): The path to the MLflow credentials file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    with open(os.path.expanduser(mlflow_credentials_file)) as f:
        for line in f:
            key, value = line.strip().split('=')
            key = key.split(' ')[-1]
            value = value.replace("'", "").replace('"', '')
            os.environ[key] = value


def get_config_from_mlflow(experiment_name, tracking_uri, run_idx=0):
    """
    Retrieves the configuration file from MLflow for a given experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        run_idx (int, optional): The index of the run to retrieve the configuration from. Defaults to 0.

    Returns:
        dict: The configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Get the experiment and enumerate the runs that have been executed
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_id)

    # Assumes first run in list is the last one that was executed
    run = runs.loc[run_idx]

    # Find all artifacts
    artifact_repo = mlflow.artifacts.get_artifact_repository(run.artifact_uri)
    all_artifacts = artifact_repo.list_artifacts()

    # find the artifacts that have the name `config_*.json`
    config_artifacts = [a for a in all_artifacts if 'config_' in a.path]
    if not config_artifacts:
        raise FileNotFoundError("Configuration file not found.")

    config = config_artifacts[-1]

    # download the config file
    temp_dir = tempfile.mkdtemp()
    artifact_repo.download_artifacts(config.path, temp_dir)
    config_path = os.path.join(temp_dir, config.path)
    with open(config_path) as f:
        config = json.load(f)
    return config


def try_get_checkpoint(
    experiment_name, 
    tracking_uri, 
    checkpoint_dir='.',
    run_idx=0
):
    """
    Try to get the latest checkpoint from the database, if that fails, get it from the local logs.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        run_idx (int, optional): The index of the run to retrieve the checkpoint from. Defaults to 0.

    Returns:
        str: The path to the latest checkpoint file.
    """
    try:
        model_weights_file = get_checkpoint_from_database(
            experiment_name,
            tracking_uri,
            run_idx,
        )
    except:
        model_weights_file = get_checkpoint_from_local_logs(
            experiment_name, 
            tracking_uri,
            checkpoint_dir,
        )
    return model_weights_file


def get_checkpoint_from_database(
    experiment_name,
    tracking_uri,
    run_idx=0,
):
    """
    Retrieves the latest checkpoint from the specified MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        run_idx (int, optional): The index of the run to retrieve the checkpoint from. Defaults to 0.

    Returns:
        str: The path to the latest checkpoint file.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_id)
    # Assumes first run in list is the last one that was executed
    run = runs.loc[run_idx]
    artifact_repo = mlflow.artifacts.get_artifact_repository(run.artifact_uri)
    all_artifacts = artifact_repo.list_artifacts()
    model_artifacts = [a for a in all_artifacts if a.path.startswith("model")]
    checkpoint_artifact = model_artifacts[-1]
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, "checkpoint")
    os.makedirs(checkpoint_dir)

    # download the checkpoint
    artifact_repo.download_artifacts(checkpoint_artifact.path, checkpoint_dir)

    # list the downloaded files including walking the directories
    downloaded_checkpoints = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".ckpt"):
                downloaded_checkpoints.append(
                    os.path.join(root, file)
                )

    latest_checkpoint = downloaded_checkpoints[-1]
    return latest_checkpoint


def get_checkpoint_from_local_logs(
    experiment_name, 
    tracking_uri, 
    log_dir,
):
    """
    Get the latest checkpoint file from the local logs directory.

    Args:
        experiment_name (str): The name of the experiment.
        tracking_uri (str): The URI of the MLflow tracking server.
        log_dir (str): The directory where the logs are stored.

    Returns:
        str: The path to the latest checkpoint file.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
 
    log_dir = f'{os.path.abspath(log_dir)}/{experiment_id}'
    checkpoints = glob(f'{log_dir}/**/*.ckpt', recursive=True)
    checkpoints = sorted(checkpoints, key=os.path.getmtime)
    resume_checkpoint = checkpoints[-1]
    return resume_checkpoint
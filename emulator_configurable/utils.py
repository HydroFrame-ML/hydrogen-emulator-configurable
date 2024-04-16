import xarray as xr
import pandas as pd
import numpy as np
import dask
import torch
import torch.nn.functional as F

import os
import mlflow
from glob import glob
from tqdm.autonotebook import tqdm
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar


dask.config.set(**{'array.slicing.split_large_chunks': True})

def spatial_gradient_penalty_loss(yhat, ytru, space_weight=1, loss_fun=F.mse_loss):
    loss = loss_fun(ytru, yhat)

    dx_tru = torch.diff(ytru, dim=-1)
    dx_hat = torch.diff(yhat, dim=-1)
    dx_loss = loss_fun(dx_tru, dx_hat)

    dy_tru = torch.diff(ytru, dim=-2)
    dy_hat = torch.diff(yhat, dim=-2)
    dy_loss = loss_fun(dy_tru, dy_hat)

    return loss + space_weight * (dx_loss + dy_loss)


def zarr_data_gen(
    files,
    selectors={},
    chunks={'x': 112, 'y':112, 'z':5, 'time': 7}
):
    ds = xr.open_mfdataset(
        files, engine='zarr', consolidated=False, data_vars='minimal'
    ).chunk(chunks).isel(**selectors)
    train_ds = ds.isel(time=slice(0, -1))
    
    # TODO : Remove this, this should be a part of the data pipeline
    depth_varying_params = ['van_genuchten_alpha',  'van_genuchten_n',  'porosity',  'permeability']
    for zlevel in range(5):
        train_ds[f'pressure_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(0, -1)).drop('time')
        train_ds[f'pressure_next_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(1, None)).drop('time')
        for v in depth_varying_params:
            train_ds[f'{v}_{zlevel}'] = ds[v].isel(z=zlevel)

    train_ds = train_ds.assign_coords({
        'time': np.arange(len(train_ds['time'])),
        'x': np.arange(len(train_ds['x'])),
        'y': np.arange(len(train_ds['y'])),
        'z': np.arange(len(ds['z'])),
    })
    return train_ds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_to_device(seq, device):
    return [s.to(device) for s in seq]


def match_dims(x, target):
    return x.reshape([len(x) if i == len(x) else 1 for i in target.shape])


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

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

def step_of_checkpoint(path):
    base = path.split('=')[-1]
    step_number = base.split('.')[0]
    return int(step_number)


def find_best_checkpoint(
    log_dir,
    experiment_name,
    uri_scheme='file:',
    uri_authority='',
):
    tracking_uri = f'{uri_scheme}{uri_authority}{log_dir}'
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = client.list_run_infos(experiment_id)
    # Assumes first run is current and second is the most recently completed
    test_run = runs[1]
    run_id = test_run.run_id
    run_path = f'{log_dir}/{experiment_id}/{run_id}'
    run_dict = mlflow.get_run(run_id).to_dictionary()
    checkpoint_dir = run_dict['data']['params']['checkpoint_dir']
    tracking_metric = 'train_loss'
    metric_df = pd.read_csv(f'{run_path}/metrics/{tracking_metric}',
                            delim_whitespace=True,
                            names=['time', tracking_metric, 'step'],
                            index_col=2)
    epoch_df = pd.read_csv(f'{run_path}/metrics/epoch',
                            delim_whitespace=True,
                            names=['time', 'epoch', 'step'],
                            index_col=2)

    best_step = metric_df[tracking_metric].idxmin()+1
    current_epoch = int(epoch_df.loc[best_step-1]['epoch'])
    checkpoint_file = f'epoch={current_epoch}-step={best_step}.ckpt'
    best_checkpoint = f'{checkpoint_dir}/{checkpoint_file}'
    return best_checkpoint


def find_all_checkpoints(
    log_dir,
    experiment_name,
    uri_scheme='file:',
    uri_authority='',
    run_idx=1,
):
    tracking_uri = f'{uri_scheme}{uri_authority}{log_dir}'
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    # Assumes first run is current and second is the most recently completed
    test_run = runs[run_idx]
    run_id = test_run.info.run_id
    run_path = f'{log_dir}/{experiment_id}/{run_id}'
    run_dict = mlflow.get_run(run_id).to_dictionary()
    checkpoint_dir = run_dict['data']['params']['checkpoint_dir']
    checkpoints = sorted(glob(f'{checkpoint_dir}/*.ckpt'))
    return checkpoints




def find_resume_checkpoint(
    log_dir,
    experiment_name,
    uri_scheme='file:',
    uri_authority='',
    run_idx=1,
):
    tracking_uri = f'{uri_scheme}{uri_authority}{log_dir}'
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    # Assumes first run is current and second is the most recently completed
    test_run = runs[run_idx]
    run_id = test_run.info.run_id
    run_path = f'{log_dir}/{experiment_id}/{run_id}'
    run_dict = mlflow.get_run(run_id).to_dictionary()
    checkpoint_dir = run_dict['data']['params']['checkpoint_dir']
    checkpoints = sorted(glob(f'{checkpoint_dir}/*.ckpt'))
    resume_checkpoint = checkpoints[-1]
    return resume_checkpoint


def find_last_checkpoint(
    experiment_name,
    tracking_uri,
):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    latest = np.argsort([r.info.start_time for r in runs])[::-1]
    try:
        test_run = runs[latest[0]]
        run_id = test_run.info.run_id
        run_path = f'{log_dir}/{experiment_id}/{run_id}'
        run_dict = mlflow.get_run(run_id).to_dictionary()
        checkpoint_dir = run_dict['data']['params']['checkpoint_dir']
        checkpoints = sorted(glob(f'{checkpoint_dir}/*.ckpt'), key=os.path.getmtime)
        last_checkpoint = checkpoints[-1]
    except:
        test_run = runs[latest[1]]
        run_id = test_run.info.run_id
        run_path = f'{log_dir}/{experiment_id}/{run_id}'
        run_dict = mlflow.get_run(run_id).to_dictionary()
        checkpoint_dir = run_dict['data']['params']['checkpoint_dir']
        checkpoints = sorted(glob(f'{checkpoint_dir}/*.ckpt'), key=os.path.getmtime)
        last_checkpoint = checkpoints[-1]
    return last_checkpoint


def find_checkpoint(
    method,
    log_dir,
    experiment_name,
    uri_scheme='file:',
    uri_authority='',
):
    pass


def get_full_metric_df(
    log_dir,
    experiment_name,
    metric_name='train_loss',
    uri_scheme='file:',
    uri_authority='',
):
    tracking_uri = f'{uri_scheme}{uri_authority}{log_dir}'
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    run_ids = [r.info.run_id for r in runs]
    start_times = [r.info.start_time for r in runs]

    sorted_idx = np.argsort(start_times)
    sorted_ids = [run_ids[i] for i in sorted_idx]
    iter_count = 0
    df_list = []
    for run_id in sorted_ids:
        run_path = f'{log_dir}/{experiment_id}/{run_id}'
        loss_file = f'{run_path}/metrics/{metric_name}'
        try:
            df = pd.read_csv(
                loss_file,
                delim_whitespace=True,
                header=None,
                index_col=2,
                names=['time',  metric_name]
            )
            df.index += iter_count
            iter_count = df.iloc[-1].name
            df_list.append(df)
        except FileNotFoundError:
            continue

    df = pd.concat(df_list)
    return df


def save_state_dict_from_checkpoint(
    log_dir,
    experiment_name,
    out_file,
    uri_scheme='file:',
    uri_authority='',
):
    ckpt_file = find_last_checkpoint(
        log_dir, experiment_name
    )
    state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))['state_dict']
    torch.save(state_dict, out_file)


def load_state_dict_from_checkpoint(
    log_dir,
    experiment_name,
    uri_scheme='file:',
    uri_authority='',
):
    ckpt_file = find_last_checkpoint(
        log_dir, experiment_name
    )
    state_dict = torch.load(
        ckpt_file,
        map_location=torch.device('cpu')
    )['state_dict']
    return state_dict
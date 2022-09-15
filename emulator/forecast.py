import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch import nn
from .models import (
    MultiStepMultiLayerModel,
    MultiLSTMModel,
    BaseLSTM,
    UNet,
    BasicConvNet,
)
from glob import glob
from .dataset import RecurrentDataset
from .model_builder import ModelBuilder
from hydroml import scalers
from hydroml.process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_surface_forecast(ds, config):
    """
    Run only the surface portion of the forecast. This
    encompasses only processes which would be simulated by CLM

    Parameters
    ----------
    ds: xr.Dataset
        The input data. This should at least have `seqence_length` elements
        in the time dimension as well as the variables required in the
        `parameters`, `in_vas`, and `state_vars` sections of the configuration.
    config: dict
        A configuration for the surface forecast. It's specs are:
        {
          'scaler_file': path to the scalers used during training,
          'parameters': parameter values used as inputs,
          'in_vars': input forcing or other time dependent variables,
          'state_vars': input state variables from the subsurface,
          'out_vars': the variables that the forecast is expect to produce,
          'sequence_length': the length of time in the input data
          'forecast_length': the length of time that the forecast produces
          'nlayers': number of hidden layers in the forecast model,
          'hidden_dim': the dimension of the hidden layers in the model,
          'model_state_file': the path to the saved, trained model
        }

    Returns
    -------
    The xarray dataset with the added `out_vars` with `forecast_length`
    elements in the time dimension
    """

    # Pull some config
    conus_scalers = scalers.load_scalers(config['scaler_file'])
    patch_sizes = {'x': len(ds['x']), 'y': len(ds['y'])}
    in_vars = config['forcing_vars']
    surface_parameters = config['surface_parameters']
    subsurface_parameters = config['subsurface_parameters']
    parameters = surface_parameters+subsurface_parameters
    state_vars = config['state_vars']
    out_vars = config['out_vars']
    seq_len = config['sequence_length']

    # Create the dataset
    dataset = RecurrentDataset(
        lambda: ds ,
        static_inputs=parameters,
        forcing_inputs=in_vars,
        state_inputs=state_vars,
        dynamic_outputs=out_vars,
        scalers=conus_scalers,
        sequence_length=seq_len,
        patch_sizes=patch_sizes,
    )
    dataset.per_worker_init()

    # Create the model
    model_config = config.get('model_def', {}).get('model_config', {})
    model_config['forcing_vars'] = config['forcing_vars']
    model_config['surface_parameters'] = config['surface_parameters']
    model_config['subsurface_parameters'] = config['subsurface_parameters']
    model_config['state_vars'] = config['state_vars']
    model_config['out_vars'] = config['out_vars']
    model_config['sequence_length'] = config['forecast_length']

    model = ModelBuilder.build_emulator(
        emulator_type=config['model_def']['type'],
        model_config=config['model_def']['model_config']
    )
    weights = torch.load(config['model_state_file'], map_location=DEVICE)
    model.load_state_dict(weights)
    model = model.to(DEVICE)
    model.eval();

    # Run the forecast
    out_vars = {v: [] for v in config['out_vars']}
    for m in range(len(ds['member'])):
        x = dataset._get_inputs(ds.isel(member=m))
        # Switch dims to be (y, x, time, var)
        xx = x.permute(2,3,0,1).to(DEVICE)
        with torch.no_grad():
            yy = torch.stack([model(xxx) for xxx in xx])
        # Switch dims back to (time, var, y, x)
        yy = yy.permute(3,2,0,1).cpu().numpy()
        for i, v in enumerate(out_vars):
            out_vars[v].append(conus_scalers[v].inverse_transform(yy[i]))

    # Assemble and return the forecast
    pred_ds = xr.Dataset()# ds.isel(time=slice(-config['forecast_length'], None))
    for v in config['out_vars']:
        pred_ds[v] = xr.DataArray(
            np.stack(out_vars[v]), dims=['member', 'time', 'y', 'x'], name=v)
    if 'swe' in pred_ds:
        pred_ds['swe'] = pred_ds['swe'].where(pred_ds['swe'] > 0, other=0.0)
    #pred_ds = pred_ds.assign_coords(ds.coords)
    return pred_ds


def run_subsurface_forecast(ds, config):
    """
    Run only the subsurface portion of the forecast. This
    encompasses only processes which would be simulated by ParFlow

    Parameters
    ----------
    ds: xr.Dataset
        The input data. This should at least have `forecast_lenght` elements
        in the time dimension as well as the variables required in the
        `parameters`, `in_vas`, and `state_vars` sections of the configuration.
    config: dict
        A configuration for the subsurface forecast. It's specs are:
        {
          'scaler_file': path to the scalers used during training,
          'parameters': parameter values used as inputs,
          'in_vars': input forcing or other time dependent variables,
          'state_vars': input state variables from the subsurface,
          'out_vars': the variables that the forecast is expect to produce,
          'forecast_length': the length of time that the forecast produces
          'model_state_file': the path to the saved, trained model
        }

    Returns
    -------
    The xarray dataset with the added `out_vars` with `forecast_length`
    elements in the time dimension
    """

    # Pull some config
    conus_scalers = scalers.load_scalers(config['scaler_file'])
    patch_sizes = {'x': len(ds['x']), 'y': len(ds['y'])}
    in_vars = config['forcing_vars']
    surface_parameters = config['surface_parameters']
    subsurface_parameters = config['subsurface_parameters']
    parameters = surface_parameters+subsurface_parameters
    state_vars = config['state_vars']
    out_vars = config['out_vars']
    seq_len = config['forecast_length']

    # Create the dataset
    dataset = RecurrentDataset(
        lambda: ds ,
        static_inputs=parameters,
        forcing_inputs=in_vars,
        state_inputs=state_vars,
        dynamic_outputs=out_vars,
        scalers=conus_scalers,
        sequence_length=seq_len,
        patch_sizes=patch_sizes,
    )
    dataset.per_worker_init()

    # Create model and process functions
    model_config = config.get('model_def', {}).get('model_config', {})
    model_config['forcing_vars'] = config['forcing_vars']
    model_config['surface_parameters'] = config['surface_parameters']
    model_config['subsurface_parameters'] = config['subsurface_parameters']
    model_config['state_vars'] = config['state_vars']
    model_config['out_vars'] = config['out_vars']
    model_config['sequence_length'] = config['forecast_length']

    model = ModelBuilder.build_emulator(
        emulator_type=config['model_def']['type'],
        model_config=config['model_def']['model_config']
    )
    weights = torch.load(config['model_state_file'], map_location=DEVICE)
    model.load_state_dict(weights)
    model = model.to(DEVICE)
    model.eval();
    sat_fun = SaturationHead()
    wtd_fun = WaterTableDepthHead()
    flow_fun = OverlandFlowHead()

    # Set up some data
    pred_ds = xr.Dataset()
    dswe = ds['swe'].diff('time')
    melt = -1 * dswe.where(dswe < 0, other=0.0)
    melt = xr.concat([xr.zeros_like(ds['swe'].isel(time=[0])), melt], dim='time')
    melt.name = 'melt'
    ds['melt'] = melt
    if 'time' not in ds['pressure'].dims:
        ds['pressure'] = ds['pressure'].expand_dims({'time': seq_len})
    ds['topographic_index'] = xr.broadcast(ds['z'], ds['topographic_index'])[1]
    alpha = torch.tensor(ds['van_genuchten_alpha'].values)
    n = torch.tensor(ds['van_genuchten_n'].values)
    slope_x = torch.tensor(ds['slope_x'].values)
    slope_y = torch.tensor(ds['slope_y'].values)
    mannings = 0.0 * torch.clone(slope_x) + 2.0 #FIXME: CHECK THIS VALUE!!

    # Run the forecast
    pressure = []
    saturation = []
    water_table_depth = []
    streamflow = []
    with torch.no_grad():
        for m in range(len(ds['member'])):
            x = dataset._get_inputs(ds.isel(member=m))[np.newaxis, ...].to(DEVICE)
            pressure.append(model(x).cpu().numpy().squeeze())
    pressure = np.stack(pressure)
    pressure = xr.DataArray(pressure, dims=['member', 'time', 'z', 'y', 'x'])
    pressure = conus_scalers['pressure'].inverse_transform(pressure)
    pred_ds['pressure'] = pressure
    #TODO: FIXME: This shouldn't be hardcoded
    dz = torch.tensor([100.0, 1.0, 0.6, 0.3, 0.1])

    # Calculate derived quantities
    for m in range(len(ds['member'])):
        pressure_tensor = torch.from_numpy(pressure.isel(member=m).values)
        saturation_tensor = sat_fun(pressure_tensor, alpha, n)
        water_table_depth_tensor = torch.stack([wtd_fun(
            pressure_tensor[i],
            saturation_tensor[i],
            dz
        ) for i in range(seq_len)])
        streamflow_tensor = torch.stack([flow_fun(
            pressure_tensor[i],
            slope_x.squeeze(),
            slope_y.squeeze(),
            mannings.squeeze(),
            1000.0, #TODO: FIXME: These shouldn't be hardcoded
            1000.0, #TODO: FIXME: These shouldn't be hardcoded
            flow_method='OverlandFlow',
        ) for i in range(seq_len)])
        saturation.append(saturation_tensor.cpu().numpy())
        water_table_depth.append(water_table_depth_tensor.cpu().numpy())
        streamflow.append(streamflow_tensor.cpu().numpy())

    # Put the data together and return it
    pred_ds['saturation'] = xr.DataArray(
        np.stack(saturation), dims=['member', 'time', 'z', 'y', 'x'])
    pred_ds['water_table_depth'] = xr.DataArray(
        np.stack(water_table_depth), dims=['member', 'time', 'y', 'x'])
    pred_ds['streamflow'] = xr.DataArray(
        np.stack(streamflow), dims=['member', 'time', 'y', 'x'])
    pred_ds['soil_moisture'] = pred_ds['saturation'] * ds['porosity']
    return pred_ds


def run_forecast(
    ds: xr.Dataset,
    land_surface_config: dict,
    subsurface_config: dict
) -> xr.Dataset:
    """
    Run the combined forecast.

    Parameters
    ----------
    ds: xr.Dataset
        The input data for the forecast
    land_surface_config: dict
        The surface forecast config. See the `run_surface_forecast`
        documentation for the specs of what goes in here.
    subsurface_config: dict
        The subsurface forecast config. See the `run_subsurface_forecast`
        documentation for the specs of what goes in here.
    """
    surf_ds = run_surface_forecast(ds, land_surface_config)
    ds = ds.isel(time=slice(-land_surface_config['forecast_length'], None))
    ds['et'] = surf_ds['et']
    ds['swe'] = surf_ds['swe']
    subsurf_ds = run_subsurface_forecast(ds, subsurface_config)
    return xr.merge([surf_ds, subsurf_ds])

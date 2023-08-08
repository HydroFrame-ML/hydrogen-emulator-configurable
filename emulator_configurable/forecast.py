import torch
import numpy as np
import pandas as pd
import xarray as xr
import hydroml as hml
from torch import nn
from glob import glob
from tqdm import tqdm
from .datapipes import create_new_loader, create_batch_generator
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
        TODO: FIXME: OUTDATED DOCSTRING

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
    forcings = config['forcings']
    parameters = config['parameters']
    states = config['states']
    targets = config['targets']
    seq_len = config['forecast_length']
    nx, ny = len(ds['x']), len(ds['y'])

    # Create model and process functions
    model = ModelBuilder.build_emulator(
        type=config['model_def']['type'],
        config=config['model_def']['config']
    )
    if 'model_state_file' in config:
        weights = torch.load(config['model_state_file'], map_location=DEVICE)
    elif 'run_name' in config:
        assert 'log_dir' in config, (
            'You put the run name, but not the location!')
        log_dir = config['log_dir']
        run_name = config['run_name']
        weights = hml.utils.load_state_dict_from_checkpoint(log_dir, run_name)
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

    vgn_a = torch.tensor(ds['van_genuchten_alpha'].values)
    vgn_n = torch.tensor(ds['van_genuchten_n'].values)
    slope_x = torch.tensor(ds['slope_x'].values)
    slope_y = torch.tensor(ds['slope_y'].values)

    #TODO: FIXME: These shouldn't be hardcoded
    mannings = 0.0 * torch.clone(slope_x) + 2.0 #FIXME: CHECK THIS VALUE!!
    dz = torch.tensor([100.0, 1.0, 0.6, 0.3, 0.1])

    # Run the forecast
    all_pres = []
    all_sat = []
    all_wtd = []
    all_flow = []

    single_run = False
    if 'member' not in ds:
        single_run = True
        ds = ds.expand_dims({'member': 1})

    with torch.no_grad():
        for m in range(len(ds['member'])):
            dataset = create_new_loader(
                ds.isel(member=m), config['scaler_file'],
                seq_len, ny, nx,
                forcings, parameters, states, targets,
                batch_size=1, num_workers=1,
                input_overlap={}, return_partial=True,
                augment=False, shuffle=False,
            )
            # Run the emulator
            for batch in dataset:
                forcing, state, params, _ = hml.utils.sequence_to_device(batch, DEVICE)
                pres_m = model(forcing, state, params)
                # Need to convert to dataset to inverse transform
                # then convert back to tensor so it can be used
                # to calculate the derived quantities, not ideal :(
                pres_m = conus_scalers['pressure'].inverse_transform(
                    xr.DataArray(pres_m.cpu().numpy().squeeze(), dims=['time', 'z', 'y', 'x'])
                ).values
                pres_m = torch.tensor(pres_m)
                all_pres.append(pres_m.cpu().numpy())

            # Calculate derived quantities
            sat = sat_fun.forward(pres_m, vgn_a, vgn_n)
            wtd = wtd_fun.forward(
                pres_m, sat, dz, depth_ax=1
            )
            flow = torch.stack([flow_fun(
                pres_m[i],
                slope_x.squeeze(),
                slope_y.squeeze(),
                mannings.squeeze(),
                1000.0, #TODO: FIXME: These shouldn't be hardcoded
                1000.0, #TODO: FIXME: These shouldn't be hardcoded
                flow_method='OverlandFlow',
            ) for i in range(pres_m.shape[0])])
            # Save the results
            all_sat.append(sat.cpu().numpy())
            all_wtd.append(wtd.cpu().numpy())
            all_flow.append(flow.cpu().numpy())

    # Put the results into a dataset
    pred_ds['pressure'] = xr.DataArray(
        np.stack(all_pres), dims=('member', 'time', 'z', 'y', 'x')
    )
    pred_ds['saturation'] = xr.DataArray(
        np.stack(all_sat), dims=('member', 'time', 'z', 'y', 'x')
    )
    pred_ds['water_table_depth'] = xr.DataArray(
        np.stack(all_wtd), dims=('member', 'time', 'y', 'x')
    )
    pred_ds['streamflow'] = xr.DataArray(
        np.stack(all_flow), dims=('member', 'time', 'y', 'x')
    )
    pred_ds['soil_moisture'] = pred_ds['saturation'] * ds['porosity']
    pred_ds = pred_ds.assign_coords(ds.coords)
    return pred_ds.squeeze()


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
    depth_varying_params = ['van_genuchten_alpha',  'van_genuchten_n',  'porosity',  'permeability']
    for zlevel in range(5):
        ds[f'pressure_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(1, -1)).drop('time')
        ds[f'pressure_next_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(2, None)).drop('time')
        ds[f'pressure_prev_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(0, -2)).drop('time')
        for v in depth_varying_params:
            ds[f'{v}_{zlevel}'] = ds[v].isel(z=zlevel)

    surf_ds = run_surface_forecast(ds, land_surface_config)
    ds = ds.isel(time=slice(-land_surface_config['forecast_length'], None))
    ds['et'] = surf_ds['et']
    ds['swe'] = surf_ds['swe']
    subsurf_ds = run_subsurface_forecast(ds, subsurface_config)
    return xr.merge([surf_ds, subsurf_ds])

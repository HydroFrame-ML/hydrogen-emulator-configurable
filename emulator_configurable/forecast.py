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
    # TODO: Write some tests around this, or maybe use a dataclass
    scaler_dict = scalers.load_scalers(config['scaler_file'])
    forcings = config['forcings']
    parameters = config['parameters']
    states = config['states']
    targets = config['targets']
    seq_len = config['forecast_length']
    nx, ny = len(ds['x']), len(ds['y'])
    input_dims = {'time': seq_len, 'y': ny, 'x': nx}

    # Create model and process functions
    model = ModelBuilder.build_emulator(
        type=config['model_def']['type'],
        config=config['model_def']['config']
    )
    if 'model_state_file' in config['model_def']:
        weights = torch.load(config['model_def']['model_state_file'], map_location=DEVICE)
    elif 'run_name' in config['model_def']:
        assert 'log_dir' in config['model_def'], (
            'You put the run name, but not the location!')
        log_dir = config['log_dir']
        run_name = config['run_name']
        weights = hml.utils.load_state_dict_from_checkpoint(log_dir, run_name)
    model.load_state_dict(weights)
    model = model.to(torch.float32).to(DEVICE)
    model.eval();
    sat_fun = SaturationHead()
    wtd_fun = WaterTableDepthHead()
    flow_fun = OverlandFlowHead()

    # Accounts for if we only give an initial condition vs the full timeseries
    if 'time' not in ds['pressure'].dims:
        ds['pressure'] = ds['pressure'].expand_dims({'time': seq_len})

    #TODO: FIXME: These shouldn't be hardcoded, put in the call to the data catalog
    if 'mannings' not in ds:
        ds['mannings'] = 0.0 * ds['slope_x'] + 2.0 
    dz = torch.tensor([100.0, 1.0, 0.6, 0.3, 0.1])
    dx, dy = 1000.0, 1000.0

    # Run the forecast
    batch_pred_list = []

    # Loader tracks the actual data
    dataset = create_new_loader(
        ds, config['scaler_file'],
        seq_len, ny, nx,
        forcings, parameters, states, targets,
        batch_size=1, num_workers=1,
        input_overlap={}, return_partial=True,
        augment=False, shuffle=False,
    )
    # Batch generator tracks the coordinates
    bgen = create_batch_generator(
        ds, 
        input_dims=input_dims, 
        input_overlap={},
        return_partial=True
    )

    # Run the emulator
    for (batch, coords) in zip(dataset, bgen):
        # Set up coordinates so we can merge stuff together easily later
        # Note we don't need z in coords because it is always fully represented in a batch
        dims= ['time', 'z', 'y', 'x']
        ymin, ymax = coords['y'].min().values[()], coords['y'].max().values[()]
        xmin, xmax = coords['x'].min().values[()], coords['x'].max().values[()]
        coords = {
            'time': coords['time'],
            'y': np.arange(ymin, ymax+1),
            'x': np.arange(xmin, xmax+1),
        }

        # Run the model
        forcing, state, params, _ = hml.utils.sequence_to_device(batch, DEVICE)
        with torch.no_grad():
            pres_tensor = model(forcing, state, params)

        # Need to convert to data_array to inverse transform & record
        # Also need to do the transpose because the inverse transform
        # may result in a different order of dimensions
        pres_da = xr.DataArray(pres_tensor.squeeze().cpu().numpy(), dims=dims, coords=coords)
        pres_da = scaler_dict['pressure'].inverse_transform(pres_da)
        pres_da = pres_da.transpose(*dims).astype(np.float32)

        # then convert back to tensor so it can be used
        # to calculate the derived quantities, not ideal :(
        pres_tensor = torch.from_numpy(pres_da.values)

        # Calculate derived quantities
        seldict = dict(x=coords['x'], y=coords['y'])
        vgn_a = torch.from_numpy(ds['van_genuchten_alpha'].sel(**seldict).values)
        vgn_n = torch.from_numpy(ds['van_genuchten_n'].sel(**seldict).values)
        slope_x = torch.from_numpy(ds['slope_x'].sel(**seldict).values)
        slope_y = torch.from_numpy(ds['slope_y'].sel(**seldict).values)
        mannings = torch.from_numpy(ds['mannings'].sel(**seldict).values)

        sat = sat_fun.forward(pres_tensor, vgn_a, vgn_n)
        wtd = wtd_fun.forward(pres_tensor, sat, dz, depth_ax=1)
        flow = torch.stack([flow_fun(
            pres_tensor[t],
            slope_x,
            slope_y,
            mannings,
            dx, dy, flow_method='OverlandFlow',
        ) for t in range(pres_tensor.shape[0])])

        # Save the results with xarray so we can put it back together later
        pred_ds = xr.Dataset()
        pred_ds['pressure'] = pres_da
        pred_ds['water_table_depth'] = xr.DataArray(
            wtd.cpu().numpy(), 
            dims=('time', 'y', 'x'), coords=coords
        )
        pred_ds['saturation'] = xr.DataArray(
            sat.cpu().numpy(), 
            dims=('time', 'z', 'y', 'x'), coords=coords
        )
        pred_ds['streamflow'] = xr.DataArray(
            flow.cpu().numpy(), 
            dims=('time', 'y', 'x'), coords=coords
        )
        batch_pred_list.append(pred_ds)

    # Put everything back together
    pred_ds = xr.combine_by_coords(batch_pred_list, coords='all')
    pred_ds['soil_moisture'] = pred_ds['saturation'] * ds['porosity']
    pred_ds = pred_ds.assign_coords(ds.coords)
    # TODO: is this squeeze needed?
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

    # TODO: This needs to go in data catalog lookup
    depth_varying_params = ['van_genuchten_alpha',  'van_genuchten_n',  'porosity',  'permeability']
    # TODO: The 5 here is hardcoded, it should be pulled from the data catalog
    for zlevel in range(5):
        ds[f'pressure_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(1, -1)).drop('time')
        ds[f'pressure_next_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(2, None)).drop('time')
        ds[f'pressure_prev_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(0, -2)).drop('time')
        for v in depth_varying_params:
            ds[f'{v}_{zlevel}'] = ds[v].isel(z=zlevel)

    surf_ds = run_surface_forecast(ds, land_surface_config)
    ds = ds.isel(time=slice(-land_surface_config['forecast_length'], None))
    # TODO: do not hardcode these
    ds['et'] = surf_ds['et']
    ds['swe'] = surf_ds['swe']
    subsurf_ds = run_subsurface_forecast(ds, subsurface_config)
    return xr.merge([surf_ds, subsurf_ds])

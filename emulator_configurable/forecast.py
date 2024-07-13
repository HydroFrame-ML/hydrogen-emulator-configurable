import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch import nn
from glob import glob
from tqdm import tqdm
from . import scalers
from .utils import sequence_to_device, load_state_dict_from_checkpoint
from .datapipes import create_new_loader, create_batch_generator
from .model_builder import ModelBuilder
from .process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_surface_forecast(ds, config):
    raise NotImplementedError("This function is not implemented yet")


def run_subsurface_forecast(
        ds, config
    ):

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

    # TODO: Need to find out how to load the saved checkpoint from the mlflow server
    if 'model_state_file' in config['model_def']:
        weights = torch.load(config['model_def']['model_state_file'], map_location=DEVICE)
    elif 'run_name' in config['model_def']:
        assert 'log_dir' in config['model_def'], (
            'You put the run name, but not the location!')
        log_dir = config['log_dir']
        run_name = config['run_name']
        weights = load_state_dict_from_checkpoint(log_dir, run_name)
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
        forcing, state, params, _ = sequence_to_device(batch, DEVICE)
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
    surf_ds = run_surface_forecast(ds, land_surface_config)
    # Pull out the last N timesteps to run the subsurface forecast
    ds = ds.isel(time=slice(-land_surface_config['forecast_length'], None))
    # TODO: do not hardcode these
    ds['et'] = surf_ds['et']
    ds['swe'] = surf_ds['swe']
    subsurf_ds = run_subsurface_forecast(ds, subsurface_config)
    return xr.merge([surf_ds, subsurf_ds])

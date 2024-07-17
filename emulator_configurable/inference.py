import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch import nn
from glob import glob
from tqdm import tqdm
from typing import List, Union, Optional

from . import scalers
from .utils import sequence_to_device, get_checkpoint_from_database
from .datapipes import create_new_loader, create_batch_generator
from .model_builder import ModelBuilder, model_setup
from .process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)


def run_surface_inference(ds, config):
    raise NotImplementedError("This function is not implemented yet")


def run_subsurface_inference(
    ds: xr.Dataset,
    model_type: str,
    model_config: dict,
    model_weights: dict,
    forcings: List[str],
    parameters: List[str],
    states: List[str],
    *,
    device: Optional[Union[torch.device, str]]='cuda',
    sequence_length: Optional[int]=None,
    patch_size: Optional[int]=None,
    targets: Optional[List[str]]=None,
    scaler_file: Optional[str]=None,
    **kwargs
):
    # Make sure we have all the necessary extra stuff in the dataset
    assert 'van_genuchten_alpha' in ds, "van_genuchten_alpha is required in the dataset"
    assert 'van_genuchten_n' in ds, "van_genuchten_n is required in the dataset"
    assert 'porosity' in ds, "porosity is required in the dataset"
    assert 'permeability' in ds, "permeability is required in the dataset"
    assert 'slope_x' in ds, "slope_x is required in the dataset"
    assert 'slope_y' in ds, "slope_y is required in the dataset"

    # Deal with optional arguments
    # Targets can be empty if we're just doing auto-regression
    # from the initial conditions found in the state variables
    if targets is None:
        targets = states
    # If sequence_length is not provided, we'll just use the full timeseries
    if sequence_length is None:
        sequence_length = len(ds['time'])
    # If patch_size is not provided, we'll use the full domain
    if patch_size:
        nx, ny = patch_size, patch_size
    else:
        nx, ny = len(ds['x']), len(ds['y'])
    # If scalers aren't specified we use the defaults built into the code
    if scaler_file is None:
        scale_dict = scalers.DEFAULT_SCALERS
    else:
        scale_dict = scalers.load_scalers(scaler_file)

    # Create model and process functions
    model = model_setup(
        model_type=model_type,
        model_config=model_config,
        model_weights=model_weights,
        precision=torch.float32,
        device=device,
    ).eval()

    sat_fun = SaturationHead()
    wtd_fun = WaterTableDepthHead()
    flow_fun = OverlandFlowHead()

    # Accounts for if we only give an initial condition vs the full timeseries
    for v in states:
        if 'time' not in ds[v].dims:
            ds[v] = ds[v].expand_dims({'time': len(ds['time'])})

    #TODO: FIXME: These shouldn't be hardcoded, put in the call to the data catalog
    if 'mannings' not in ds:
        ds['mannings'] = 0.0 * ds['slope_x'] + 2.0 
    dz = torch.tensor([100.0, 1.0, 0.6, 0.3, 0.1]).to(device)
    dx, dy = 1000.0, 1000.0

    # Loader tracks the actual data
    dataset = create_new_loader(
        ds, 
        nt=sequence_length,
        ny=ny, 
        nx=nx,
        forcings=forcings, 
        parameters=parameters, 
        states=states, 
        targets=targets,
        batch_size=1, 
        num_workers=1,
        input_overlap={}, 
        return_partial=True,
        augment=False, 
        shuffle=False,
    )

    # Batch generator tracks the coordinates
    input_dims = {'time': sequence_length, 'y': ny, 'x': nx}
    bgen = create_batch_generator(
        ds, 
        input_dims=input_dims, 
        input_overlap={},
        return_partial=True
    )

    # Run the emulator
    batch_pred_list = []
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
        forcing, state, params, _ = sequence_to_device(batch, device)
        with torch.no_grad():
            # predictions.shape is (batch, time, channel, y, x)
            # but we squeeze out the batch dimension because we only have one
            pressure = model(forcing, state, params).squeeze()

        # Unscale the predictions
        for i, v in enumerate(targets):
            pressure[:, i, ...] = scale_dict[v].inverse_transform(pressure[:, i, ...])

        # Calculate derived quantities
        seldict = dict(x=coords['x'], y=coords['y'])
        vgn_a = torch.from_numpy(ds['van_genuchten_alpha'].sel(**seldict).values).to(device)
        vgn_n = torch.from_numpy(ds['van_genuchten_n'].sel(**seldict).values).to(device)
        slope_x = torch.from_numpy(ds['slope_x'].sel(**seldict).values).to(device)
        slope_y = torch.from_numpy(ds['slope_y'].sel(**seldict).values).to(device)
        mannings = torch.from_numpy(ds['mannings'].sel(**seldict).values).to(device)

        sat = sat_fun.forward(pressure, vgn_a, vgn_n)
        wtd = wtd_fun.forward(pressure, sat, dz, depth_ax=1)
        flow = torch.stack([flow_fun(
            pressure[t],
            slope_x,
            slope_y,
            mannings,
            dx, dy, flow_method='OverlandFlow',
        ) for t in range(pressure.shape[0])])

        # Save the results with xarray so we can put it back together later
        pred_ds = xr.Dataset()
        pred_ds['pressure'] = xr.DataArray(
            pressure.cpu().numpy(),
            dims=dims, coords=coords
        )
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
    return pred_ds


def run_inference(
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
    surf_ds = run_surface_inference(ds, land_surface_config)
    # Pull out the last N timesteps to run the subsurface forecast
    ds = ds.isel(time=slice(-land_surface_config['forecast_length'], None))
    # TODO: do not hardcode these
    ds['et'] = surf_ds['et']
    ds['swe'] = surf_ds['swe']
    subsurf_ds = run_subsurface_inference(ds, subsurface_config)
    return xr.merge([surf_ds, subsurf_ds])

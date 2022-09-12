import xarray as xr
import pandas as pd
import numpy as np
import dask

dask.config.set(**{'array.slicing.split_large_chunks': True})
NLDAS_PFMETA_PATH = '/hydrodata/forcing/processed_data/CONUS1/NLDAS2/daily/'
PFCLM_PFMETA_PATH = '/hydrodata/PFCLM/CONUS1_baseline/simulations/daily/'

def conus1_data_gen(selectors={}):
    dask.config.set(scheduler='single-threaded')
    ds = xr.open_mfdataset([
        f'{NLDAS_PFMETA_PATH}/conus1_nldas_daily_2003.pfmetadata',
        f'{NLDAS_PFMETA_PATH}/conus1_nldas_daily_2004.pfmetadata',
        f'{NLDAS_PFMETA_PATH}/conus1_nldas_daily_2005.pfmetadata',
        f'{PFCLM_PFMETA_PATH}/conus1_pfclm_daily_2003.pfmetadata',
        f'{PFCLM_PFMETA_PATH}/conus1_pfclm_daily_2004.pfmetadata',
        f'{PFCLM_PFMETA_PATH}/conus1_pfclm_daily_2005.pfmetadata',
        '/hydrodata/PFCLM/CONUS1_baseline/simulations/static/conus1_parameters.pfmetadata'
    ],  chunks={'time': 4, 'x': 500, 'y': 500, 'z': 5})
    ds = ds.isel(**selectors)
    ds['depth'] = xr.DataArray([0.1, 0.3, 0.6, 1.0, 100.0][::-1], dims=['z'])
    ds['topographic_index'] = xr.broadcast(ds['z'], ds['topographic_index'])[1]
    dswe = ds['swe'].diff('time')
    melt = -1 * dswe.where(dswe < 0, other=0.0)
    melt.name = 'melt'
    ds['melt'] = melt

    pressure_1 = ds['pressure'].isel(time=slice(1, None))
    pressure_1 = pressure_1.assign_coords(coords={'time': pressure_1['time'] - pd.Timedelta('1D')})
    ds = ds.isel(time=slice(0, -1))
    ds['pressure_1'] = pressure_1
    ds = ds.assign_coords({
        'time': np.arange(len(ds['time'])),
        'x': np.arange(len(ds['x'])),
        'y': np.arange(len(ds['y'])),
        'z': np.arange(len(ds['z'])),
    })
    return ds.astype(np.float32).squeeze()


def zarr_data_gen(selectors={}):
    ds = xr.open_dataset(
        '/home/SHARED/data/ab6361/conus1_2006_preprocessed.zarr',
        engine='zarr', consolidated=False
    )
    ds = ds.assign_coords({
        'time': np.arange(len(ds['time'])),
        'x': np.arange(len(ds['x'])),
        'y': np.arange(len(ds['y'])),
        'z': np.arange(len(ds['z'])),
    })
    return ds.squeeze()


def netcdf_data_gen(selectors={}):
    dask.config.set(scheduler='single-threaded')
    ds = xr.open_mfdataset(['/home/SHARED/data/ab6361/conus1_2006_test.nc',
                            '/home/SHARED/data/ab6361/conus1_2006_met_test.nc',])
    #ds = xr.open_dataset(file)
    ds = ds.isel(**selectors)

    pressure_next = ds['pressure'].isel(time=slice(1, None))
    pressure_next = pressure_next.assign_coords(coords={
        'time': pressure_next['time'] - pd.Timedelta('1D')
    })
    dswe = ds['swe'].diff('time')
    melt = -1 * dswe.where(dswe < 0, other=0.0)
    melt.name = melt
    ds = ds.isel(time=slice(0, -1))
    ds['pressure_next'] = pressure_next
    ds['melt'] = melt

    ds = ds.assign_coords({
        'time': np.arange(len(ds['time'])),
        'x': np.arange(len(ds['x'])),
        'y': np.arange(len(ds['y'])),
        'z': np.arange(len(ds['z'])),
    }).chunk({"time": -1})
    ds = ds.astype(np.float32)
    if 'vegtype' in ds:
        ds['vegtype'] = ds['vegetation_type'].astype(np.int64)
    return ds


def layered_data_gen(selectors={}):
    ds = xr.open_dataset('/home/SHARED/data/ab6361/conus1_2006_test.nc')
    met_ds = xr.open_dataset('/home/SHARED/data/ab6361/conus1_2006_met_test.nc')
    ds = ds.update(met_ds)
    ds = ds.isel(**selectors)
    train_ds = ds.isel(time=slice(0, -1))
    train_ds = train_ds.drop(['time', 'latitude', 'longitude'])
    depth_varying_params = ['van_genuchten_alpha',  'van_genuchten_n',  'porosity',  'permeability']

    for zlevel in range(5):
        train_ds[f'pressure_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(0, -1)).drop('time')
        train_ds[f'pressure_next_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(1, None)).drop('time')
        for v in depth_varying_params:
            train_ds[f'{v}_{zlevel}'] = ds[v].isel(z=zlevel)
    train_ds['melt'] = ds['swe'].diff(dim='time').drop('time')

    train_ds = train_ds.assign_coords({
        'time': np.arange(len(train_ds['time'])),
        'x': np.arange(len(train_ds['x'])),
        'y': np.arange(len(train_ds['y'])),
        #'z': np.arange(len(ds['z'])),
    })
    return train_ds


def adjacent_layer_indices(layer_number, max_layers=4):
    if layer_number == 0:
        needed_layers = [layer_number + i for i in [0, 1]]
    elif layer_number == max_layers:
        needed_layers = [layer_number + i for i in [-1, 0]]
    else:
        needed_layers = [layer_number + i for i in [-1, 0, 1]]
    return needed_layers


def adjacent_layer_varlist(layer_number, varlist, max_layers=4):
    out_list = []

    if layer_number == 0:
        needed_layers = [layer_number + i for i in [0, 1]]
    elif layer_number == max_layers:
        needed_layers = [layer_number + i for i in [-1, 0]]
    else:
        needed_layers = [layer_number + i for i in [-1, 0, 1]]

    for v in varlist:
        for l in needed_layers:
            out_list.append(f'{v}_{l}')

    return out_list

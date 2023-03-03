import torch
import hydroml as hml
import xarray as xr
import xbatcher as xb

from functools import partial
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from torch.utils.data import DataLoader

@functional_datapipe("xbatcher")
class XbatcherDataPipe(IterDataPipe):
    def __init__(self, parent_pipe, input_dims, **kwargs):
        self.parent_pipe = parent_pipe
        self.input_dims = input_dims
        self.kwargs = kwargs

    def __iter__(self):
        for dataarray in self.parent_pipe:
            bgen = xb.BatchGenerator(dataarray, self.input_dims, **self.kwargs)
            for batch in bgen:
                yield batch

    def __len__(self):
        bgens = [xb.BatchGenerator(ds, self.input_dims, **self.kwargs) for ds in self.parent_pipe]
        return sum(len(bgen) for bgen in bgens)


class OpenDatasetPipe(IterDataPipe):
    def __init__(self, file_list, varlist, iselectors={}):
        super().__init__()
        self.ds = self.open_files(file_list).isel(**iselectors)[varlist]

    def open_files(self, files):
        ds = xr.open_mfdataset(files, engine='zarr', compat='override', coords='minimal')
        train_ds = ds.isel(time=slice(1, -1))
        depth_varying_params = ['van_genuchten_alpha',  'van_genuchten_n',  'porosity',  'permeability']
        for zlevel in range(5):
            train_ds[f'pressure_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(1, -1)).drop('time')
            train_ds[f'pressure_next_{zlevel}'] = ds['pressure'].isel(z=zlevel, time=slice(2, None)).drop('time')
            for v in depth_varying_params:
                train_ds[f'{v}_{zlevel}'] = ds[v].isel(z=zlevel)
        return train_ds

    def __iter__(self):
        yield self.ds


def transform(batch, scalers):
    dims_and_coords = set(batch.dims).union(set(batch.coords))
    variables = set(batch.variables)
    for k in variables - dims_and_coords:
        batch[k] = scalers[k].transform(batch[k])
    return batch


def batch_time(batch, nt):
    return (batch.coarsen(time=nt)
                 .construct(time=('batch', 'ts'))
                 .drop('time')
                 .rename({'ts': 'time'}))


def augment(batch):
    if torch.rand(1) > 0.5:
        batch = batch.isel(x=slice(None, None, -1))
    if torch.rand(1) > 0.5:
        batch = batch.isel(y=slice(None, None, -1))
    return batch


def load(batch):
    return batch.load()


def split_and_convert(
    batch,
    layer_parameters,
    layer_states,
    layer_forcings,
    layer_targets,
    dtype
):
    lp = layer_parameters
    ls = layer_states
    lf = layer_forcings
    lt = layer_targets
    forcing = batch[lf].to_array().transpose('batch', 'time', 'variable', 'y', 'x')
    params = (batch[lp].isel(time=[0]).to_array()
                       .transpose('batch', 'time', 'variable', 'y', 'x'))
    state = (batch[ls].isel(time=[0]).to_array()
                      .transpose('batch', 'time', 'variable', 'y', 'x'))
    target = batch[lt].to_array().transpose('batch', 'time', 'variable', 'y', 'x')

    forcing = torch.tensor(forcing.values).to(dtype)
    params = torch.tensor(params.values).to(dtype)
    state = torch.tensor(state.values).to(dtype)
    target = torch.tensor(target.values).to(dtype)
    return forcing, state, params, target


def create_new_loader(
    files,
    scaler_file,
    nt,
    ny,
    nx,
    forcings,
    parameters,
    states,
    targets,
    batch_size,
    num_workers,
    selectors={},
    dtype=torch.float32,
):
    time_size = nt * batch_size
    dataset_files = files
    scalers = hml.scalers.load_scalers(scaler_file)
    input_dims = {'time': time_size, 'y': ny, 'x': nx}
    input_overlap = {'time': time_size//4, 'y': ny//4, 'x': nx//4}

    # Partial function application
    batch_nt = partial(batch_time, nt=nt)
    transform_fn = partial(transform, scalers=scalers)
    convert = partial(
        split_and_convert,
        layer_parameters=parameters,
        layer_states=states,
        layer_forcings=forcings,
        layer_targets=targets,
        dtype=dtype
    )

    pipe = OpenDatasetPipe(
        dataset_files,
        forcings+parameters+states+targets,
        iselectors=selectors
    )
    pipe = pipe.xbatcher(
        input_dims=input_dims,
        input_overlap=input_overlap,
        preload_batch=False,
        shuffle=True
    )
    steps_per_epoch = len(pipe)
    pipe = pipe.map(batch_nt)
    pipe = pipe.map(augment)
    pipe = pipe.map(load)
    pipe = pipe.map(transform_fn)
    pipe = pipe.map(convert)

    dl = DataLoader(
        pipe,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    return dl, steps_per_epoch

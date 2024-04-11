import dask
import torch
import scalers
import numpy as np
import xarray as xr
import xbatcher as xb
import parflow as pf

# WARNING: Hack for certain versions of torch/torchdata
torch.utils.data.datapipes.utils.common._check_lambda_fn = None

from functools import partial
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from torch.utils.data import DataLoader


def slices_to_icoords(selector):
    return {k: np.arange(v.start, v.stop, v.step)
            for k, v in selector.items()}


def create_batch_generator(
    files_or_ds, input_dims, selectors={}, **kwargs
):
    if isinstance(files_or_ds, xr.Dataset):
        ds = files_or_ds.isel(**selectors)
    else:
        ds = open_files(files_or_ds, selectors)
    dims = dict(ds.dims)
    shape = tuple(dims.values())
    dummy_ds = xr.DataArray(np.empty(shape), dims=dims, coords=ds.coords)
    bgen = xb.BatchGenerator(dummy_ds, input_dims, **kwargs)
    return bgen


def estimate_xbatcher_pipe_size(
    files, selectors,
    input_dims, **kwargs
):
    bgen = create_batch_generator(
        files, selectors=selectors,
        input_dims=input_dims, **kwargs
    )
    return len(bgen)


def open_files(files, selectors, var_list=None, load=False):
    ds = xr.open_mfdataset(files, engine='zarr', compat='override', coords='minimal').chunk({'time': 24})
    ds = ds.assign_coords({
        'x': np.arange(len(ds['x'])),
        'y': np.arange(len(ds['y']))
    }).isel(**selectors)
    train_ds = ds
    stack_vars = {
        'van_genuchten_alpha': [f'van_genuchten_alpha_{i}' for i in range(5)],
        'van_genuchten_n': [f'van_genuchten_n_{i}' for i in range(5)],
    }
    for k, v in stack_vars.items():
        train_ds[k] = xr.concat([ds[i] for i in v], dim='z')

    if var_list:
        train_ds = train_ds[var_list]
    if load:
        train_ds = train_ds.load()
    return train_ds


@functional_datapipe("xbatcher")
class XbatcherDataPipe(IterDataPipe):
    def __init__(self, parent_pipe, input_dims,  number_batches, **kwargs):
        self.parent_pipe = parent_pipe
        self.input_dims = input_dims
        self.number_batches = number_batches
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
    def __init__(self, files_or_ds, var_list=None, selectors={}, nthreads=8, load=False):
        super().__init__()
        if isinstance(files_or_ds, xr.Dataset):
            self.ds = files_or_ds.isel(**selectors)
            if var_list:
                ds = ds[var_list]
            if load:
                self.ds = self.ds.load()
        else:
            self.file_list = files_or_ds
            self.ds = None
        self.var_list = var_list
        self.selectors = selectors
        self.nthreads = nthreads
        self.load = load

    def per_worker_init(self):
        from multiprocessing.pool import ThreadPool
        import dask
        dask.config.set(pool=ThreadPool(self.nthreads))
        self.ds = open_files(self.file_list, self.selectors, self.var_list, load=self.load)

    def __iter__(self):
        # Call this only when we enter the iterator, ensuring
        # every worker has it's own thread pool, hopefully
        if not self.ds: self.per_worker_init()
        yield self.ds


def add_feature_txt(
    batch,
    file='/hydrodata/PFCLM/CONUS1_baseline/other_domain_files/CONUS.pitfill.txt',
    name='elevation'
):
    data = np.loadtxt(file, skiprows=1).reshape(1888, 3342)
    data = xr.DataArray(data, dims=('y', 'x'))
    batch[name] = data.isel(x=slice(0, len(batch['x'])), y=slice(0, len(batch['y'])))
    return batch

def add_feature_pfb(
    batch,
    file='/home/ab6361/hydrogen_workspace/data/Frac_dist_100_withfilter.pfb',
    name='frac_dist'
):
    data = pf.read_pfb(file).squeeze()
    data = xr.DataArray(data, dims=('y', 'x'))
    batch[name] = data.isel(x=slice(0, len(batch['x'])), y=slice(0, len(batch['y'])))
    return batch


def select_vars(batch, variables):
    return batch[variables]


def drop_coords(batch):
    return batch.drop_vars(batch.coords)


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


def concat_batch(batch):
    batch = xr.concat(batch, dim='batch')
    return batch


def augment(batch):
    if torch.rand(1) > 0.5:
        batch = batch.isel(x=slice(None, None, -1))
    if torch.rand(1) > 0.5:
        batch = batch.isel(y=slice(None, None, -1))
    return batch


def load(batch):
    return batch.compute()


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
    dims = ('batch', 'time', 'variable', 'y', 'x')
    forcing = batch[lf].to_array().transpose(*dims)
    try:
        params = batch[lp].isel(time=[0]).to_array().transpose(*dims)
    except:
        params = batch[lp].expand_dims({'time': 1}).isel(time=[0]).to_array()
        params = params.transpose(*dims)
    state = batch[ls].isel(time=[0]).to_array().transpose(*dims)
    target = batch[lt].to_array().transpose(*dims)

    forcing = torch.tensor(forcing.values).to(dtype)
    params = torch.tensor(params.values).to(dtype)
    state = torch.tensor(state.values).to(dtype)
    target = torch.tensor(target.values).to(dtype)
    return forcing, state, params, target


def torch_concat(batch):
    return [torch.cat(b) for b in batch]


def load_in_parallel(batch):
    def _single_load(sample):
        return sample.load()
    batch = dask.compute([dask.delayed(_single_load)(sample) for sample in batch])[0]
    return batch


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
    input_overlap=None,
    return_partial=False,
    augment=False,
    shuffle=True,
    selectors={},
    dtype=torch.float32,
    pin_memory=True,
    persistent_workers=True,
):
    dataset_files = files
    scalers = scalers.load_scalers(scaler_file)
    scalers['cbrt_swe']      = scalers.get('cbrt_swe', scalers.MinMaxScaler(0, 3))
    scalers['cbrt_swe_prev'] = scalers['cbrt_swe']
    scalers['cbrt_swe_next'] = scalers['cbrt_swe']
    scalers['rainfall'] = scalers['APCP']
    for i in range(5):
        scalers[f'pressure_prev_{i}'] = scalers[f'pressure_{i}']
        scalers[f'pressure_next_{i}'] = scalers[f'pressure_{i}']
    scalers[f'swe_prev'] = scalers[f'swe']
    scalers[f'et_prev'] = scalers[f'et']

    input_dims = {'time': nt, 'y': ny, 'x': nx}
    if input_overlap is None:
        input_overlap = {'time': nt//4, 'y': ny//3, 'x': nx//3}

    number_batches = estimate_xbatcher_pipe_size(
        files=dataset_files,
        selectors=selectors,
        input_dims=input_dims,
        input_overlap=input_overlap,
        return_partial=return_partial,
        preload_batch=False
    )

    # Partial function application
    sel_vars = partial(select_vars, variables=forcings+parameters+states+targets)
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
        selectors=selectors
    )
    pipe = pipe.map(sel_vars)
    pipe = pipe.xbatcher(
        input_dims=input_dims,
        input_overlap=input_overlap,
        number_batches=number_batches,
        preload_batch=False,
        return_partial=return_partial,
        shuffle=shuffle,
    )
    pipe = pipe.sharding_filter()
    pipe = pipe.map(drop_coords)
    pipe = pipe.batch(batch_size)
    pipe = pipe.map(load_in_parallel)
    pipe = pipe.map(concat_batch)
    if augment:
        pipe = pipe.map(augment)
    pipe = pipe.map(transform_fn)
    pipe = pipe.map(convert)
    pipe.number_batches = number_batches
    dl = DataLoader(
        pipe,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=False,
    )
    dl.number_batches = number_batches
    return dl

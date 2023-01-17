import itertools
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import IterableDataset, Dataset, DataLoader, random_split
from typing import Optional, Union, List, Mapping


def worker_init_fn(worker_id):
    """
    Helper function to initialize a worker in a pytorch DataLoader
    This basically just gets the underlying Dataset to call `per_worker_init`,
    which is implemented by all of the Dataset classes in this module.

    Parameters
    ----------
    worker_id:
        The worker id (not used, but required by the DataLoader interface)
    """
    # get_worker_info returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset  # The Dataset copy in this worker process.
    dataset_obj.per_worker_init(n_workers=worker_info.num_workers)


class RecurrentDataset(IterableDataset):
    """
    The RecurrentDataset is a class for loading spatio-temporal data.
    There are several "categories" of data which can be loaded by this
    class. First is "static inputs" - these variables are those which are
    spatial only in nature, and have no time dimension. They are loaded and
    stacked together as a copy in time to appear as though they have temporal
    information. Next is "forcing inputs" - these are considered spatiotemporal,
    but do not contain any depth, so the space component is 2 dimensional.
    Next, are "state inputs", which are full 3d in space and 1d in time. This
    generally is considered for subsurface variables which are simulated by
    Parflow. Finally, "dynamic outputs" are the targets that will be returned.
    By design these are similar in nature to the "state inputs".

    The data is loaded "patch-wise", which means that a small section is
    is randomly taken from the full (y,x) dimension with a user-defined shape.
    The spatial extent can be chosen with the `patch_sizes` argument, while the
    temporal extent can be set withthe `sequence_length` argument.

    Parameters
    ----------
    data_gen_function: callable
        This is a function which returns an xarray Dataset containing
        the data which will be loaded. This must be a function so that
        it can be called by subprocess workers to allow for loading data
        in parallel using the pytorch DataLoader API. If you just want to
        pass in an existing dataset named `ds` you can usie `lambda: ds`
    static_inputs: List[str]
        A list of static input variable names to pull out from the xr.Dataset
        produced by the `data_gen_function`.
    forcing_inputs: List[str]
        A list of forcing variable names to pull out from the xr.Dataset
        produced by the `data_gen_function`.
    state_inputs: List[str]
        A list of state variable names to pull out from the xr.Dataset
        produced by the `data_gen_function`.
    dynamic_outputs: List[str]
        A list of output variable names to pull out from the xr.Dataset
        produced by the `data_gen_function`.
    sequence_length: int
        The number of timesteps per sample
    patch_sizes: Mapping[str, int]
        A mapping between dimension and number of grid cells along that
        dimension per sample (example `{'x': 25, 'y': 25}`) will pull 25x25
        patches)
    patch_strides: Mapping[str, int]
        Minimum distance between samples, this is used to reduce similarity
        between samples so that per-sample information is higher.
    dtype:
        Data type to return data as
    scalers: Mapping[str, object]
        Mapping between variable type and an instantiation of a scaling class.
        This is used to transform data from it's raw range to something roughly
        on the order [-1, 1].
    samples_per_epoch_total: int
        Number of samples in an epoch. This is used since the dataset's total
        length is not precomputed.
    initialize: bool
        Whether to initialize the dataset by calling `per_worker_init`. This
        should only be set to `True` if you are using this class without
        a DataLoader
    """

    def __init__(
        self,
        data_gen_function,
        static_inputs,
        forcing_inputs,
        state_inputs,
        dynamic_outputs,
        sequence_length,
        patch_sizes,
        patch_strides=None,
        dtype=torch.float32,
        scalers=None,
        samples_per_epoch_total=10_000,
        initialize=False,
    ):

        super().__init__()
        self.static_inputs = static_inputs
        self.forcing_inputs = forcing_inputs
        self.state_inputs = state_inputs
        self.dynamic_outputs = dynamic_outputs
        self.all_vars = (
            self.static_inputs
            + self.forcing_inputs
            + self.state_inputs
            + self.dynamic_outputs
        )

        self.scalers = scalers
        self.data_gen_function = data_gen_function
        self.dtype = dtype
        self.sequence_length = sequence_length
        self.seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=self.seed)
        self.n_samples_per_epoch_total = samples_per_epoch_total
        self.patch_sizes = patch_sizes
        if patch_strides:
            self.patch_strides = patch_strides
        else:
            self.patch_strides = patch_sizes
        if initialize:
            self.per_worker_init()

    def per_worker_init(self, n_workers=1):
        """
        Initialize the instance for use by setting up
        the samples per worker, random number generator,
        and patches.
        """
        self.ds = self.data_gen_function()
        self.n_samples_per_epoch_per_worker = (
                self.n_samples_per_epoch_total // n_workers)
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
        self.n_timesteps = len(self.ds['time'])
        if self.patch_sizes:
            self.patches = self._gen_patches()
            self.n_patches = len(self.patches)
        else:
            # Single patch that covers the entire domain
            # TODO: Handle z?
            self.patches = [{'x': slice(0, None), 'y': slice(0, None)}]
            self.n_patches = 1

    def _gen_patches(self):
        """
        Compute the lookup of spatial patches to sample from.
        """
        patch_groups = {}
        for k in self.patch_sizes.keys():
            size = self.patch_sizes[k]
            stride = self.patch_strides[k]
            patch_groups[k] = (self.ds[k].rolling(**{k: size})
                                         .construct('_', stride=stride)
                                         .dropna(k)
                                         .values)

        # Could this be more efficient?
        patches = []
        for sub_patch in itertools.product(*patch_groups.values()):
            _patch = {}
            for i, k in enumerate(self.patch_sizes.keys()):
                _patch[k] = sub_patch[i]#.astype(int)
            patches.append(_patch)
        return patches

    def _sample_a_patch(self, idx):
        """
        Select a single patch from the patch list. Assumes `_gen_patches`
        has been called previously.
        """
        if self.rng:
            patch_idx = self.rng.integers(low=0, high=len(self.patches))
            patch = self.patches[patch_idx]
        else:
            patch_idx = idx % self.n_patches
            patch = self.patches[patch_idx]
        return patch

    def augment(self, ds):
        """
        Perform some data augmentation by random transposes and flips
        """
        if not self.rng:
            return ds
        if self.rng.random() > 0.25:
            ds = ds.transpose(..., 'x', 'y')
        if self.rng.random() > 0.25:
            ds = ds.reindex(x=ds['x'][::-1])
        if self.rng.random() > 0.25:
            ds = ds.reindex(y=ds['y'][::-1])
        return ds

    def get_sample_explicit(self, selector):
        """
        Explicitly grab a sample from a given selector that
        goes into the `ds.sel` method from xarray.
        """
        sample = self.ds.isel(selector).load()
        return self._get_sample(sample)

    def __iter__(self):
        for i in range(self.n_samples_per_epoch_per_worker):
            patch = self._sample_a_patch(i)
            time_slice = self._sample_time_slice(i)
            selector = {**patch, **time_slice}
            sample = self.ds.sel(selector).load()
            sample = self.augment(sample)
            x, y = self._get_sample(sample)
            yield x, y

    def _sample_time_slice(self, idx):
        """
        Sample a time slice. The time slice produced will be of length
        `sequence_length`.
        """
        max_time = len(self.ds['time']) - self.sequence_length + 1
        if self.rng:
            idx = self.rng.integers(low=0, high=max_time)
        time_slice = {'time': slice(idx, idx+self.sequence_length-1)}
        return time_slice

    def _get_inputs(self, sel_ds=None):
        """
        Pull out the input data, scale it, and then stack it together
        as a torch tensor
        """
        if not sel_ds:
            sel_ds = self.ds
        sel_ds = sel_ds[self.forcing_inputs
                        + self.static_inputs
                        + self.state_inputs]
        sel_ds = sel_ds.load()
        sel_ds = sel_ds.ffill(dim='time').bfill(dim='time')
        X = []
        for v in self.forcing_inputs:
            if v in self.scalers:
                scaled_x = np.array(self.scalers[v].transform(sel_ds[v]))
            else:
                scaled_x = sel_ds[v].values
            if len(scaled_x.shape) == 3:
                scaled_x = scaled_x.reshape(
                    self.sequence_length, 1, *scaled_x.shape[-2:])
            X.append(scaled_x)

        for v in self.static_inputs:
            if v in self.scalers:
                scaled_x = np.array(self.scalers[v].transform(sel_ds[v]))
            else:
                scaled_x = sel_ds[v].values
            if len(scaled_x.shape) == 2:
                scaled_x = scaled_x.reshape(-1, *scaled_x.shape)
            scaled_x = np.stack([scaled_x for _ in range(self.sequence_length)])
            X.append(scaled_x)

        for v in self.state_inputs:
            if v in self.scalers:
                scaled_x = self.scalers[v].transform(sel_ds[v]).values
            else:
                scaled_x = sel_ds[v].values
            if len(scaled_x.shape) == 3:
                # dims should be (time, var, y, x)
                scaled_x = scaled_x.reshape(
                    self.sequence_length, 1, *scaled_x.shape[-2:])
            X.append(scaled_x)
        #X = torch.from_numpy(np.concatenate(X, axis=1)).type(self.dtype).squeeze()
        X = torch.from_numpy(np.hstack(X)).type(self.dtype)#.squeeze()
        #if torch.sum(torch.isnan(X)):
        #    raise ValueError(f'Error in {sel_ds}')
        return X

    def _get_targets(self, sel_ds=None):
        """
        Pull out the target data, scale it, and then stack it together
        as a torch tensor
        """
        if not sel_ds:
            sel_ds = self.ds
        sel_ds = sel_ds[self.dynamic_outputs].load()
        sel_ds = sel_ds.ffill(dim='time').bfill(dim='time')
        y = []
        for v in self.dynamic_outputs:
            if v in self.scalers:
                scaled_y = self.scalers[v].transform(sel_ds[v]).values
            else:
                scaled_y = sel_ds[v].values
            if len(scaled_y.shape) == 3:
                scaled_y = scaled_y.reshape(
                    self.sequence_length, 1, *scaled_y.shape[-2:])
            y.append(scaled_y)
        y = torch.from_numpy(np.hstack(y)).type(self.dtype).squeeze()
        #if torch.sum(torch.isnan(y)):
        #    raise ValueError(f'Error in {sel_ds}')
        return y

    def _get_sample(self, sel_ds):
        """Get a single sample as defined by the sel_ds"""
        X = self._get_inputs(sel_ds)
        y = self._get_targets(sel_ds)
        return X, y




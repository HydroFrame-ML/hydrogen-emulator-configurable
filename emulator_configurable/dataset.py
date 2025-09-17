import dask
import math
import time
import torch
import numpy as np
import xarray as xr
import xbatcher as xb
import parflow as pf

from . import scalers
from functools import partial
from threading import Thread, Event, Lock
from queue import Queue
from collections import deque
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Iterator
from dask.distributed import as_completed

def open_files(files, selectors, var_list=None, load=False):
    ds = xr.open_mfdataset(files, engine='zarr', compat='override', coords='minimal', chunks='auto')
    ds = ds.assign_coords({
        'x': np.arange(len(ds['x'])),
        'y': np.arange(len(ds['y']))
    }).isel(**selectors)

    # TODO: This shouldn't be done here?
    # Or maybe it just needs to be reworked to include a calculation of the number of layers
    # and then the number of layers can be passed to the BatchGenerator
    # This is currently done here so that the resulting dataset is able to have both the 
    # variables broken out by z layer or stacked, and that is needed to complete the 
    # hybrid modeling task as opposed to the original emulator.
    # But it also seems necessary for the inference task, so maybe rethink this?
    stack_vars = {
        'van_genuchten_alpha': [f'van_genuchten_alpha_{i}' for i in range(5)],
        'van_genuchten_n': [f'van_genuchten_n_{i}' for i in range(5)],
        'porosity': [f'porosity_{i}' for i in range(5)],
    }
    for k, v in stack_vars.items():
        ds[k] = xr.concat([ds[i] for i in v], dim='z')

    if var_list:
        ds = ds[var_list]

    if load:
        ds = ds.load()
    return ds


class HydrogenDataset(Dataset):

    def __init__(
        self,
        files_or_ds,
        nt,
        ny,
        nx,
        forcings,
        parameters,
        states,
        targets,
        input_overlap=None,
        return_partial=False,
        augment=False,
        shuffle=True,
        selectors={},
        dtype=torch.float32,
        scaler_file=None,     
    ):
        """
        Pytorch dataset for loading and processing ParFlow CLM data.

        Args:
            files_or_ds (list or xarray.Dataset): List of file paths or an xarray
                Dataset containing the data.
            nt (int): Number of time steps to select for each sample.
            ny (int): Number of grid points in the y direction for each patch.
            nx (int): Number of grid points in the x direction for each patch.
            forcings (list): List of variable names to use as forcings.
            parameters (list): List of variable names to use as static parameters.
            states (list): List of variable names to use as initial conditions.
            targets (list): List of variable names to use as targets.
            input_overlap (dict, optional): Overlap sizes for each dimension.
                Defaults to None, which sets 75% time overlap and 66% spatial
                overlap.
            return_partial (bool, optional): Whether to return partial patches
                at the edges of the domain. Defaults to False.
            augment (bool, optional): Whether to apply data augmentation (random
                flips). Defaults to False.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            selectors (dict, optional): Dictionary of slices to select subsets of
                the data. Defaults to {} (selects all data).
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to
                torch.float32.
            scaler_file (str, optional): Path to a file containing scalers for
                each variable. If None, default scalers will be used. Defaults to
                None.
        """
        super().__init__()

        self.files = files_or_ds if not isinstance(files_or_ds, xr.Dataset) else None
        self.selectors = selectors
        self.nt = nt
        self.ny = ny
        self.nx = nx
        if input_overlap is None:
            self.input_overlap = {'time': (3*nt)//4, 'y': (2*ny)//3, 'x': (2*nx)//3}
        else:
            self.input_overlap = input_overlap

        self.return_partial = return_partial
        self.shuffle = shuffle
        
        self.forcings = forcings
        self.parameters = parameters
        self.states = states
        self.targets = targets
        self.vars_of_interest = forcings + parameters + states + targets
        
        self.augment = augment
        self.dtype = dtype

        if scaler_file is None:
            self.scale_dict = scalers.DEFAULT_SCALERS
        else:
            self.scale_dict = scalers.load_scalers(scaler_file)

        input_dims = {'time': nt, 'y': ny, 'x': nx}
        
        if isinstance(files_or_ds, xr.Dataset):
            self.ds = files_or_ds.isel(**selectors)
        else:
            self.ds = open_files(files_or_ds, selectors)

        bgen = xb.BatchGenerator(
            self.ds, 
            input_dims=input_dims, 
            input_overlap=input_overlap, 
            return_partial=return_partial, 
            shuffle=shuffle
        )
        self.length = len(bgen)


    def __len__(self):
        return self.length
    

    def transform(self, batch):
        dims_and_coords = set(batch.dims).union(set(batch.coords))
        variables = set(batch.variables)
        for k in variables - dims_and_coords:
            batch[k] = self.scale_dict[k].transform(batch[k])
        return batch


    def augment_batch(self, batch):
        if torch.rand(1) > 0.5:
            batch = batch.copy().isel(x=slice(None, None, -1))
        if torch.rand(1) > 0.5:
            batch = batch.copy().isel(y=slice(None, None, -1))
        return batch


    def split_and_convert(
        self,
        batch,
        layer_parameters,
        layer_states,
        layer_forcings,
        layer_targets,
    ):
        lp = layer_parameters
        ls = layer_states
        lf = layer_forcings
        lt = layer_targets
        dims = ('time', 'variable', 'y', 'x')
        
        # Prepare all lazy operations (no computation yet)
        forcing_lazy = batch[lf].to_array().transpose(*dims)
        state_lazy = batch[ls].isel(time=[0]).to_array().transpose(*dims)
        target_lazy = batch[lt].to_array().transpose(*dims)
        
        # Handle params with special case (but still keep it lazy)
        try:
            params_lazy = batch[lp].isel(time=[0]).to_array().transpose(*dims)
        except:
            params_lazy = batch[lp].expand_dims({'time': 1}).isel(time=[0]).to_array().transpose(*dims)
        
        # Single batched compute operation for all variables
        # This is the key optimization - compute everything at once
        forcing_computed, params_computed, state_computed, target_computed = dask.compute(
            forcing_lazy, params_lazy, state_lazy, target_lazy,
            scheduler='threads'  # Use threads instead of synchronous for better performance
        )
        
        # Convert to PyTorch tensors
        forcing = torch.tensor(forcing_computed.values).to(self.dtype)
        params = torch.tensor(params_computed.values).to(self.dtype)
        state = torch.tensor(state_computed.values).to(self.dtype)
        target = torch.tensor(target_computed.values).to(self.dtype)
        
        return forcing, state, params, target


    def init_worker(self):
        if self.files is not None:
            self.ds = open_files(self.files, self.selectors)
        else:
            self.ds = self.files_or_ds.isel(**self.selectors)
        
        input_dims = {'time': self.nt, 'y': self.ny, 'x': self.nx}
        self.bgen = xb.BatchGenerator(
            self.ds,
            input_dims=input_dims,
            input_overlap=self.input_overlap,
            return_partial=self.return_partial,
            shuffle=self.shuffle
        )


    def __getitem__(self, idx):
        start_time = time.time()

        # Setup for multiprocessing
        if not hasattr(self, '_worker_ds'):
            init_start = time.time()
            self.init_worker()
        
        batch_gen_start = time.time()
        batch = self.bgen[idx]

        prep_start = time.time()
        if 'mannings' not in batch:
            batch['mannings'] = xr.zeros_like(batch['elevation']) + 2.0
        if self.augment:
            batch = self.augment_batch(batch)

        # Batch compute additional data as well
        # Prepare lazy operations for additional data
        additional_start = time.time()
        additional_lazy = {
            'vgn_a': batch['van_genuchten_alpha'].transpose('time', 'z', 'y', 'x'),
            'vgn_n': batch['van_genuchten_n'].transpose('time', 'z', 'y', 'x'),
            'slope_x': batch['slope_x'].transpose('time', 'y', 'x'),
            'slope_y': batch['slope_y'].transpose('time', 'y', 'x'),
            'mannings': batch['mannings'].transpose('time', 'y', 'x'),
        }
        
        # Compute all additional data at once
        additional_computed = dask.compute(additional_lazy, scheduler='threads')[0]
        
        # Convert to tensors
        additional_data = {
            key: torch.tensor(arr.values, dtype=self.dtype) 
            for key, arr in additional_computed.items()
        }

        transform_start = time.time()
        batch = batch[self.vars_of_interest]
        batch = self.transform(batch)
        
        split_start = time.time()
        forcing, state, params, target = self.split_and_convert(
            batch,
            self.parameters,
            self.states,
            self.forcings,
            self.targets,
        )
        
        return forcing, state, params, target, additional_data


class BatchedDatasetIterator:
    """
    Iterator for the BatchedDataset class. Used to manage the state of
    the dataset during iteration.
    """
    def __init__(self, dataset: 'BatchedDataset'):
        self.dataset = dataset
        self.current = 0
        self.dataset.start()
    
    def __iter__(self) -> Iterator:
        return self
    
    def __next__(self):
        if self.current >= len(self.dataset):
            self.dataset.stop()
            raise StopIteration
        
        item = self.dataset[self.current]
        self.current += 1
        return item


class BatchedDataset(Dataset):
    """
    A dataset wrapper that provides batched access to another dataset,
    with prefetching and parallel data loading using Dask.
    """
    def __init__(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        client, 
        prefetch_factor: int = 2,
        drop_last: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.client = client
        self.drop_last = drop_last
        
        # Calculate number of batches
        self.num_samples = len(dataset)
        self.num_batches = math.floor(self.num_samples / batch_size)
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1
            
        # Use a deque for accumulating samples for the current batch
        self.current_batch = deque(maxlen=batch_size)
        self.current_batch_lock = Lock()
        
        # Queue for completed batches
        self.batches_queue = Queue(maxsize=prefetch_factor)
        self.stop_event = Event()
        
        # Keep track of pending futures
        self.pending_futures = set()
        self.futures_lock = Lock()
        
        # Track current position
        self.current_idx = 0
        
        # Initialize and start the prefetch thread
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
    def __iter__(self):
        return BatchedDatasetIterator(self)
        
    def start(self):
        """Ensure the prefetch thread is running"""
        if not self.prefetch_thread.is_alive():
            self.stop_event.clear()
            self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()
            
    def stop(self):
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.stop_event.set()
            self.prefetch_thread.join(timeout=5.0)
            # Clear queues and pending futures
            self.current_batch.clear()
            with self.futures_lock:
                self.pending_futures.clear()
            while not self.batches_queue.empty():
                try:
                    self.batches_queue.get_nowait()
                except:
                    pass

    def __len__(self):
        return self.num_batches
    
    def _collate_fn(self, batch):
        forcings = torch.stack([sample[0] for sample in batch])
        states = torch.stack([sample[1] for sample in batch])
        params = torch.stack([sample[2] for sample in batch])
        targets = torch.stack([sample[3] for sample in batch])
        extras = {
            key: torch.stack([sample[-1][key] for sample in batch]) 
            for key in batch[0][-1].keys()
        }
        return forcings, states, params, targets, extras

    def _get_next_indices(self, current_idx: int):
        """Calculate the next set of indices to fetch"""
        start_idx = current_idx * self.batch_size
        # Calculate how many samples we need to fill the prefetch buffer
        samples_needed = min(
            self.batch_size * self.prefetch_factor - self.batches_queue.qsize(),
            self.num_samples - start_idx
        )
        if samples_needed <= 0:
            return []
        return list(range(start_idx, start_idx + samples_needed))

    def _submit_new_futures(self):
        """Submit new futures if we need more"""
        with self.futures_lock:
            if len(self.pending_futures) < self.batch_size * self.prefetch_factor:
                # Get next batch of indices
                indices = self._get_next_indices(self.current_idx)
                if indices:
                    # Submit tasks to dask
                    futures = self.client.map(
                        self.dataset.__getitem__,
                        indices,
                        pure=False
                    )
                    self.pending_futures.update(futures)
                    self.current_idx = (self.current_idx + 1) % len(self)
                    return True
            return False

    def _prefetch_worker(self):
        #dask.config.set(scheduler='synchronous')
        dask.config.set({'logging.distributed': 'error'})
        while not self.stop_event.is_set():
            try:
                # Submit new futures if needed
                self._submit_new_futures()
                
                # Process completed futures
                with self.futures_lock:
                    if not self.pending_futures:
                        time.sleep(0.01)
                        continue
                    
                    # Create iterator for completed futures
                    future_iterator = as_completed(self.pending_futures, with_results=True)
                
                # Process each completed future as it arrives
                for future, result in future_iterator:
                    if self.stop_event.is_set():
                        break
                        
                    # Remove from pending set
                    with self.futures_lock:
                        self.pending_futures.remove(future)
                    
                    # Add to current batch
                    with self.current_batch_lock:
                        self.current_batch.append(result)
                        
                        # If we have a full batch, collate and move to batches queue
                        if len(self.current_batch) == self.batch_size:
                            batch = list(self.current_batch)
                            self.current_batch.clear()
                            collated_batch = self._collate_fn(batch)
                            self.batches_queue.put(collated_batch, timeout=5.0)
                            
                    # Submit new futures if needed
                    self._submit_new_futures()
                    
                # Handle partial batch if needed
                if not self.drop_last:
                    with self.current_batch_lock:
                        if len(self.current_batch) > 0 and not self.pending_futures:
                            batch = list(self.current_batch)
                            self.current_batch.clear()
                            collated_batch = self._collate_fn(batch)
                            self.batches_queue.put(collated_batch, timeout=5.0)
                            
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                time.sleep(0.1)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Batch index {idx} out of range")

        try:
            # Get next batch from the batches queue
            batch = self.batches_queue.get()
            return batch
        except Exception as e:
            raise RuntimeError(f"Failed to get batch: {e}")
    
    def __del__(self):
        self.stop()

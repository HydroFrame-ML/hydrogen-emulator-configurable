import xarray as xr
import os
import torch
# Need to use xbatcher from: https://github.com/arbennett/xbatcher/tree/develop
 # See readme for installation instrutions 
import xbatcher as xb
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from parflow.tools.io import read_pfb

from torch.utils.data import Dataset


class ParFlowDataset(Dataset):

    def __init__(
        self, data_dir, run_name,
        parameter_list, patch_size, overlap, 
        param_nlayer, n_evaptrans=0, dtype=torch.float64
    ):
        super().__init__() 
        self.base_dir = f'{data_dir}/{run_name}'
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer #number of layers to use for each param, 0= use all, -n = n top layers, +n = n bottom layers
        self.patch_size = patch_size
        self.n_evaptrans = n_evaptrans
        self.overlap = overlap
        self.dtype = dtype

        self.pressure_files = sorted(glob(f'{self.base_dir}/transient/pressure*.pfb')) 
        self.pressure_files = {
            't': self.pressure_files[0:-1],
            't+1': self.pressure_files[1:]
        }
    
        self.size_test = read_pfb(self.pressure_files['t'][0])
        self.X_EXTENT = self.size_test.shape[2] 
        self.Y_EXTENT = self.size_test.shape[1]
        self.Z_EXTENT = self.size_test.shape[0]
        self.T_EXTENT = len(self.pressure_files['t'])
      
        # Create a dummy dataset that will be used to pull indices for reading subsets of the data
        self.dummy_data = xr.Dataset().assign_coords({
            'time': np.arange(self.T_EXTENT),
            'z': np.arange(self.Z_EXTENT),
            'y': np.arange(self.Y_EXTENT),
            'x': np.arange(self.X_EXTENT)
        })
   
        self.bgen = xb.BatchGenerator(
            self.dummy_data,
            input_dims={'x': self.patch_size, 'y': self.patch_size, 'time': 1},
            input_overlap={'x': self.overlap, 'y': self.overlap},
            return_partial=False,
            shuffle=True,
        )

        self.generate_namelist()

    def generate_namelist(self):
        """
        Generate a list of names that will be used to input to the model.
        This will be used as a way to record the order that the variables
        go into the model so that they can be scaled internally. See 
        model.scale_pressure, model.cale_evaptrans, and model.scale_statics
        for more information.
        """
        self.PRESSURE_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]
        self.EVAPTRANS_NAMES = [f'evaptrans_{i}' for i in range(self.n_evaptrans)]
        self.PARAM_NAMES = []
        self.OUTPUT_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]

        # Use a tiny key just to look up what we need
        patch_keys = {'x': {'start': 0, 'stop': 2},
                      'y': {'start': 0, 'stop': 2},}
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name=f'{self.base_dir}/static/{parameter}.pfb'

            # param_temp shape is (n_layers, y, x)
            param_temp = read_pfb(file_name, keys=patch_keys)

            if param_temp.shape[0] == 1:
                self.PARAM_NAMES.append(parameter)
            else: 
                temp_namelist = [f'{parameter}_{i}' for i in range(param_temp.shape[0])]
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    temp_namelist = temp_namelist[0:n_lay]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    temp_namelist = temp_namelist[n_lay:]

                # Add the new names to the list
                for t in temp_namelist:
                    self.PARAM_NAMES.append(t)
        
    def __len__(self):
        return len(self.bgen) 
    
    def __getitem__(self, idx):
        sample_indices = self.bgen[idx]

        # Pulling the indices we need
        time_index = sample_indices['time'].values[0]
        x_min, x_max = sample_indices['x'].values[[0, -1]]
        y_min, y_max = sample_indices['y'].values[[0, -1]]

        # Setting up the keys dictionary
        patch_keys = {
            'x': {'start': x_min, 'stop': x_max+1},
            'y': {'start': y_min, 'stop': y_max+1},
        }
    
        # Construct the state data and scale it:
        file_to_read = self.pressure_files['t'][time_index]
        state_data = read_pfb(file_to_read, keys=patch_keys)

        # Construct the target data and scale it:
        file_to_read_target = self.pressure_files['t+1'][time_index]
        target_data = read_pfb(file_to_read_target, keys=patch_keys)

        # Construct the parameter data and scale it:
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name=f'{self.base_dir}/static/{parameter}.pfb'
            param_temp = read_pfb(file_name, keys=patch_keys)

            if param_temp.shape[0] > 1:
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]

            parameter_data.append(param_temp)

        # Concatenate the parameter data together
        # End result is a dims of (n_parameters, y, x)
        parameter_data = np.concatenate(parameter_data, axis=0)

        #Construct the evaptrans data and scale it
        file_name_et=file_to_read.replace('pressure', 'evaptrans')
        evaptrans = (read_pfb(file_name_et, keys= patch_keys))
        #Grab the top n bottom or top layers if specified in the param_nlayer list
        #Grab the bottom n_lay layers
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        #Grab the top n_lay layers
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        
        # Convert everything to torch tensors
        state_data = torch.from_numpy(state_data).to(self.dtype)
        evaptrans = torch.from_numpy(evaptrans).to(self.dtype)
        parameter_data = torch.from_numpy(parameter_data).to(self.dtype)
        target_data = torch.from_numpy(target_data).to(self.dtype)
        return state_data, evaptrans, parameter_data, target_data
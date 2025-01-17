import xarray as xr
import os
# Need to use xbatcher from: https://github.com/arbennett/xbatcher/tree/develop
 # See readme for installation instrutions 
import xbatcher as xb
import numpy as np
import matplotlib.pyplot as plt
import scalers 

from glob import glob
from parflow.tools.io import read_pfb
#from scalers import DEFAULT_SCALERS

from torch.utils.data import Dataset

def custom_collate(batch):
    x = torch.stack(b[0] for b in batch)
    y = torch.stack(b[1] for b in batch)
    return x, y 

class ParFlowDataset(Dataset):

    def __init__(
        self, data_dir, run_name,
        parameter_list, patch_size, overlap, scaler_yaml,
        param_nlayer, n_evaptrans=0
        
    ):
        super().__init__() 
        self.base_dir = f'{data_dir}/{run_name}'
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer #number of layers to use for each param, 0= use all, -n = n top layers, +n = n bottom layers
        self.patch_size = patch_size
        self.n_evaptrans = n_evaptrans
        self.overlap = overlap
        self.scaler_yaml = scaler_yaml
        self.scaler = scalers.create_scalers_from_yaml(scaler_yaml)
        #self.scaler = DEFAULT_SCALERS

        self.pressure_files = sorted(glob(f'{self.base_dir}/transient/pressure*.pfb')) 
        self.pressure_files = {
            't': self.pressure_files[0:-1],
            't+1': self.pressure_files[1:]
        }
    
        self.size_test = read_pfb(self.pressure_files['t'][0])
        self.X_EXTENT = self.size_test.shape[2] 
        self.Y_EXTENT = self.size_test.shape[1]
        self.Z_EXTENT = self.size_test.shape[0]
        self.T_EXTENT = 1 #Change this to the number of timesteps -- should this be input up top?
      
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
            return_partial=True,
            shuffle=True,
        )

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
        for k in range(self.Z_EXTENT):
            state_data[k]= self.scaler[f'press_diff_{k}'].transform(state_data[k])

        # Construct the target data and scale it:
        file_to_read_target = self.pressure_files['t+1'][time_index]
        target_data = read_pfb(file_to_read_target, keys=patch_keys)
        for k in range(self.Z_EXTENT):
            target_data[k]= self.scaler[f'press_diff_{k}'].transform(target_data[k])

        # Construct the parameter data and scale it:
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name=f'{self.base_dir}/static/{parameter}.pfb'
            param_temp = read_pfb(file_name, keys=patch_keys)

            #Scale the data
            if param_temp.shape[0] == 1:
                param_temp = self.scaler[f'{parameter}'].transform(param_temp)
            else: 
                for k in range(param_temp.shape[0]):
                    param_temp[k]= self.scaler[f'{parameter}_{k}'].transform(param_temp[k])

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
        for k in range(self.Z_EXTENT):
            evaptrans[k]= self.scaler[f'evaptrans_{k}'].transform(evaptrans[k])
        #Grab the top n bottom or top layers if specified in the param_nlayer list
        #Grab the bottom n_lay layers
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        #Grab the top n_lay layers
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        
        # Concatenate the state data with the parameter data
        # End result is a dims of (sum(n_parameters*param_nlayer) + n_evaptrans + nz, y, x) 
        state_data = np.concatenate([state_data, evaptrans, parameter_data], axis=0)


        return state_data, target_data
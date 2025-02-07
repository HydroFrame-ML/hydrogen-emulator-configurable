# 1-Timestep emulator

This folder contains the scripts for the one time step emulator. The goal of this emulator is to predict the next timestep for a ParFlow simulation. 

## Key files in this folder: 
- `example_config.yaml`: Contains all of the settings needed for a run. These are the knobs that should be turned.
- `default_scalers....yaml`: Has the mean and standard deviation for every layer of every variable. Right now everything is set to `standard_scaler` meaning that variables will be scaled by subtracting the mean and dividing by the standard
    - `default_scalers_original`: Has the original scalers calculated by the scripts in the `CONUS2_Data_Prep` folder.
    - `default_scalers_adjusted`: changed the standard deviation to 1 for all of the layers where the standard devaition was 0 or something <1e-15.
- `main.py`: this is the main script that does the training. It call all the other classes and functions.

## Before you start: 
- In order to run a training run you first need to generate a set of test data. You can do that using `notebooks/make_subset_domain*.ipynb`
- You will also need to adjust the `example.config.yaml` to reflect your local paths and run names.
    - **Note**: The`in_channels` should equal the total number of layers you are using from your parameter list + n_evaptrans (# of evaptrans layers being used) + 10 (#of layers in a perssure file). (For example, if your input parameter list is slopex, slopey and permeabilityx and you use all layers from these files and have 4 evaptrans layers  then the in_channels will be 1+1+10+4+10 = 26)
- You may need to adjust the `default_scalers.yaml` file but these are caculated for CONUS2 so they should be good to use as is and don't need to be adjusted every single time. 

## Setup on Verde
To run on verde: 
1. Lauch an interactive *Jupyter Lab* session and selece `hydrogen-shared` as the anaconda version.

*To run python scripts:* 
1. Start a terminal session from jupyter lab (`file/new/termnial`)
2. If the terminal prompt says `(base) (hydrogen-shared)` you will need to deactivate the base enviroment with `conda deactivate` you should then just see the prompt say `(hydrogen-shared)`

*To run Jupyter Notebooks:* 
Select `Python3` as the kernel and then you shoudl be good to run

*Note:* From other enviroments you can also use `module load hydrogen-shared` to load this enviroment. 

## How to run a traning run
From terminal: `python main.py --config example_config.yaml --mode train`

## CONUS2.1 Update progress: 
- Done: 
    - Root fracs need have been re-calculated for 5 layers and with the new landcover map
    - Evaptrans have been claculated using this root zone mapping for CONUS2.1 WY2003 V2
      
- In progress:
    - Re-calculating the scalers (Evaptrans and pressure done, waiting on the new slopes and alpah and n to do the static scalers for CONUS2.1)
    - Calculting pressure scalers based on actual values not the pressure differences.
    - Creating a new `notebooks/make_subset_domain*.ipynb` that can subset from CONUS2.1 and that has rectangular rather than square patches (should be 63 by 67)
      
## Debugging next steps
1. Investigate why the initial losses are so high: Adding more time steps cut the initial loss down from 1e12 to 1e9, getting rid of all the tiny standar deviations made no difference and cutting out all the constant layers also made no difference. Next step: print mean and stdev values from the subest domain and see how they compare to the scalers file. Goal- We would like to have the initial loss down to ~1
2. Once the initail looses are more reasonable we will need to setup a testing run too. A good first test would be the same locaton but a different point in time (we have little expectation that it will do good on a different location just yet since we are training on a very small subset)

## Other things to add/change: 
1. Change the inputs so the number of layers used and the parameter list is a dictionary and not two separate lists.
2. Make a copy of the config file where the model is saved for documentation purposes. 
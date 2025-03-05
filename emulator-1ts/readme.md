# 1-Timestep emulator

This folder contains the scripts for the one time step emulator. The goal of this emulator is to predict the next timestep for a ParFlow simulation. 

## Key files in this folder: 
- `example_config.yaml`: Contains all of the settings needed for a run. These are the knobs that should be turned.
- `main.py`: this is the main script that does the training. It call all the other classes and functions.
- `*_scalers*.yaml`: Are the scalers files with the mean and standard deviation for every layer of every variable. Right now everything is set to `standard_scaler` meaning that variables will be scaled by subtracting the mean and dividing by the standard deviation. All scalers were calculated using scripts in the `CONUS2_Data_Prep` folder, refer to the readme there for more details. There are multiple versions of the scalers files available using the following naming convention: 
    - `*original` or `adjusted` : The original calcualted scalers with no adjustments made. The adjusted files have the standard deviation changed to 1 for all of the layers where the standard devaition was 0 or something <1e-15.
    - `*_pressure.yaml`: The _pressure versions have pressure scalers calcualted based on the  pressure files themselves instead of pressure file differences between timesteps. *NOTE* These are still called 'press_diff' scalers in the yaml file to match whats expected in line 90 of 'dataset.py` where the scaling happens. Should make this an option later.
    - `CONUS21_*` or `default_*`: The CONUS21 versions have the evaptrans and pressure scalers calculated from the CONUS2.1 run WY2003V2. The `default` versions use the CONUS2 Baseline run for the evaptrans and pressure scalers. 

## Before you start: 
- In order to run a training run you first need to generate a set of test data. You can do that using `notebooks/make_subset_domain*.ipynb`
- You will also need to adjust the `example.config.yaml` to reflect your local paths and run names.
    - **Note**: The`in_channels` should equal the total number of layers you are using from your parameter list + n_evaptrans (# of evaptrans layers being used) + 10 (#of layers in a perssure file). (For example, if your input parameter list is slopex, slopey and permeabilityx and you use all layers from these files and have 4 evaptrans layers  then the in_channels will be 1+1+10+4+10 = 26)


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
- New scalers have been calculated for the CONUS2.1 run and are available in this folder.
- The notebooks folder has a subset domain routine which is ready to use`make_subset_domain_CONUS21.ipynb` but before it will work WY2003V2 needs to be added to the data catalog and then the `transient_dataset` name will need to be changed to point to this. 

## Other things to add/change: 
1. We need to setup a testing run. A good first test would be the same locaton but a different point in time (we have little expectation that it will do good on a different location just yet since we are training on a very small subset)
1. Change the inputs so the number of layers used and the parameter list is a dictionary and not two separate lists.
2. Make a copy of the config file where the model is saved for documentation purposes. 
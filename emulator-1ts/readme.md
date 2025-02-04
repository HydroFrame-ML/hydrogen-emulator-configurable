# 1-Timestep emulator

This folder contains the scripts for the one time step emulator. The goal of this emulator is to predict the next timestep for a ParFlow simulation. 

## Key files in this folder: 
- `example_config.yaml`: Contains all of the settings needed for a run. These are the knobs that should be turned.
- `default_scalers....yaml`: Has the mean and standard deviation for every layer of every variable. Right now everything is set to `standard_scaler` meaning that variables will be scaled by subtracting the mean and dividing by the standard
    - `default_scalers_original`: Has the original scalers calculated by the scripts in the `CONUS2_Data_Prep` folder.
    - `default_scalers_adjusted`: changed the standard deviation to 1 for all of the layers where the standard devaition was 0 or something <1e-15.
- `main.py`: this is the main script that does the training. It call all the other classes and functions.

## Before you start: 
- In order to run a training run you first need to generate a set of test data. You can do that using `notebooks/make_subset_domain.ipynb`
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

## Next steps: 
1. Redo the CONUS2 data processing for CONUS2.1 (Scripts int `CONUS2_Data_Prep` folder):
    - The root fracs need to be re-calculated for 5 layers and with the new landcover map
    - Evaptrans files need to be created for the entire run that we will be subsetting from.
    - Scaling factors should be recalcualted (although these likely wont change much).
2.  Update the paths for subsetting in  `notebooks/make_subset_domain.ipynb` so that its using the latest CONUS2.1 2003 run for training
    - Change patches to be nx=63 by ny=67  (CONUS2.Process.Topology.P = 70
CONUS2.Process.Topology.Q = 48
CONUS2.Process.Topology.R = 1
CONUS2.ComputationalGrid.NX = 4442
CONUS2.ComputationalGrid.NY = 3256
CONUS2.ComputationalGrid.NZ = 10)
4. Investigate why the initial losses are so high: Adding more time steps cut the initial loss down from 1e12 to 1e9, getting rid of all the tiny standar deviations made no difference and cutting out all the constant layers also made no difference. Next step: print mean and stdev values from the subest domain and see how they compare to the scalers file. Goal- We would like to have the initial loss down to ~1
5. Once the initail looses are more reasonable we will need to setup a testing run too. A good first test would be the same locaton but a different point in time (we have little expectation that it will do good on a different location just yet since we are training on a very small subset)
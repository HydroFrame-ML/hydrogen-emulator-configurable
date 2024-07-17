# ParFlow Emulator for the HydroGEN project

## Environment instructions
At the moment creating and installing everything into a working environment takes a bit of work. These are the steps that should get you up and running. 

```
# Base install stuff
conda create -n hydrogen
conda activate hydrogen
conda install -c conda-forge python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge mamba
mamba install -c conda-forge torchdata pytorch-lightning xarray  xskillscore tqdm dill numcodecs dask netcdf4 zarr fsspec aiohttp pooch rioxarray holoviews geoviews jupyter jupyterlab mlflow ipykernel seaborn numcodecs pip pandas

pip install pftools

# Install the custom xbatcher version
git clone git@github.com:arbennett/xbatcher.git
cd xbatcher
git checkout cmip_swe
pip install -e .
cd ..

# Now install this package
git clone git@github.com:HydroFrame-ML/hydrogen-emulator-configurable.git
cd hydrogen-emulator-configurable
git checkout feature/new_data_and_model_flow
pip install -e .
cd ..
```

## Training models
The models are trained via the command line interface defined in `main.py`. The main entry point will be the `parflow_emulator` command. For more information about how to use the command line you can use `parflow_emulator --help`. 

You can also see some examples of training scripts in the `train_scripts` directory. 

## Inference from trained models
To run inference you will have to use the interfaces defined in the `inference.py` module.
This can be done either from the command line or from a python script. 

### Commamnd line example
```
#!/bin/bash
source ~/.mlflow_credentials
CONFIG=$(cat <<- EOM
{
    "inference_dataset_files": [
        "/hydrodata/PFCLM/CONUS1_baseline/simulations/daily/zarr/conus1_2005_preprocessed.zarr"
    ],
    "run_name": "testing_inference",
    "forcings": ["APCP", "Temp_max", "Temp_min", "melt", "et"],
    "parameters": [
        "topographic_index",
        "permeability_0",
        "permeability_1",
        "permeability_2",
        "porosity_0",
        "porosity_1",
        "porosity_2",
        "van_genuchten_alpha_0",
        "van_genuchten_alpha_1",
        "van_genuchten_alpha_2",
        "van_genuchten_n_0",
        "van_genuchten_n_1",
        "van_genuchten_n_2"
    ],
    "states": [
        "pressure_0",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "pressure_4"
    ],
    "num_workers": 8,
    "model_type": "ForcedSTRNN",
    "model_config": {
        "num_layers": 2,
        "num_hidden": [16, 16],
        "img_channel": 5,
        "out_channel": 5,
        "act_channel": 5,
        "init_cond_channel": 5,
        "static_channel": 13
    },
    "save_path": "./test.zarr"
}
EOM
)
echo $CONFIG > config_fstr_inference.json
parflow_emulator --mode predict --domain subsurface --config config_fstr_inference.json
```

### Python script example
```
from emulator_configurable.main import predict_subsurface
from emulator_configurable.utils import (
    load_mlflow_credentials,
    get_config_from_mlflow,
    update_config_for_inference,
)

# read mlflow credentials from ~/.mlflow_credentials
# which just looks like
#    export MLFLOW_TRACKING_USERNAME='username'
#    export MLFLOW_TRACKING_PASSWORD='password'
mlflow_credentials_file = '~/.mlflow_credentials'
load_mlflow_credentials(mlflow_credentials_file)

# specify experiment to pull from and logging uri
# optionally specify the run index to pull from
# where 0 just means the most recent one
experiment_name = 'testing_inference'
logging_uri = 'https://concord.princeton.edu/mlflow/'
run_idx = 0

# Pull the configuration artifact that was used to train the model
config = get_config_from_mlflow(experiment_name, logging_uri, run_idx)

# Update the config to reflect the inference settings
# These are the three things that can be supplied
# 1. (required) inference_dataset_files: list of zarr files to use for inference
#
# 2. (optional) save_path: where to save the inference results. 
#                          if not provided just return results without saving
#                          if ends with .nc save netcdf, if ends with .zarr save zarr
# 
# 3. (optional) selectors: dictionary of slices to use for inference
#                          if not provided, use the entire dataset
#                          this is useful for testing
inference_dataset_files = [
    '/hydrodata/PFCLM/CONUS1_baseline/simulations/daily/zarr/conus1_2005_preprocessed.zarr'
]
save_path = './test.nc'
selectors = {'time': slice(0, 30), 'x': slice(0,256), 'y': slice(0,256)}

# make some other minor modifications to the config for inference
config = update_config_for_inference(
    config, inference_dataset_files, save_path, selectors
)

# run the inference - note this always returns the predicted dataset
pred_ds = predict_subsurface(config)
pred_ds

# could also open the saved location with
#     import xarray as xr
#     ds = xr.open_dataset(config['save_path'])
```
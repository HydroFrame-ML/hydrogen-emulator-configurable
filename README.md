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
mamba install -c conda-forge torchdata pytorch-lightning xarray  xskillscore tqdm dill numcodecs dask netcdf4 zarr fsspec aiohttp pooch rioxarray holoviews geoviews jupyter jupyterlab mlflow ipykernel seaborn numcodecs pip pandas==1.5.2
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
The models are trained via the command line interface defined in `cli.py`. The main entry point will be the `parflow_emulator` command. For more information about how to use the command line you can use `parflow_emulator --help`. 

You can also see some examples of training scripts in the `train_scripts` directory. 

## Inference from trained models
For the moment the command line for running a trained model is not set up. To run inference you will have to use the interfaces defined in the `forecast.py` module.
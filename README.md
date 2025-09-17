# ParFlow Emulator for the HydroGEN project

## Environment instructions
The recommended way to install the environment is via [uv](https://uv.readthedocs.io/en/latest/).
The following commands will create a virtual environment, activate it, and install the required dependencies.

```
uv venv
source .venv/bin/activate
uv sync
uv pip install "git+https://github.com/arbennett/xbatcher/@cmip_swe"

# Checkout whatever branch you want to use
git checkout develop
uv pip install -e .
```

## Training models
The models are trained via the command line interface defined in `main.py`. The main entry point will be the `parflow_emulator` command. For more information about how to use the command line you can use `parflow_emulator --help`. 

You can also see some examples of training scripts in the `train_scripts` directory. 

## Inference from trained models
Inference is currently not supported via the command line interface, but is coming soon.

### Training usage example
For an example of how to train a model, see the `fstr_train_scripts` directory. Here is an example of how to train a model using a configuration file:

```bash
parflow emulator --mode train --config path/to/your/config.json
```

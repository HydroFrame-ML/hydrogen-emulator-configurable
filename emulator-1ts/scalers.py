import os
import pickle
import yaml
import utils

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCALER_PATH = f'{HERE}/default_scalers_adjusted_pressure.yaml'

def create_scalers_from_yaml(file):
    with open(file, 'r') as f:
        lookup = yaml.load(f, Loader=yaml.FullLoader)
    scalers = {}
    for k, v in lookup.items():
        scalers[k] = (float(v['mean']), float(v['std']))
    return scalers


DEFAULT_SCALERS = create_scalers_from_yaml(DEFAULT_SCALER_PATH)
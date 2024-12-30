import os
import numpy as np
import scipy.stats
import torch
import xarray as xr
import pickle
import dill
import yaml
import torch.nn.functional as F
#from . import utils
import utils

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCALER_PATH = f'{HERE}/default_scalers.yaml'


def from_pickle(pickled_dict):
    return {k: cls(**kwargs) for k, (cls, kwargs) in pickled_dict.items()}


def save_scalers(scaler_values, path):
    scaler_dict = {}
    for name, scaler in scaler_values.items():
        scaler_dict[name] = (scaler.__class__, vars(scaler))
    f = open(path, 'wb')
    dill.dump(scaler_dict, f)
    f.close()


def load_scalers(path):
    f = open(path, 'rb')
    scaler_values = dill.load(f)
    f.close()
    return from_pickle(scaler_values)


class BaseScaler:
    """ Implements base methods that all scalers should inherit """

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class OneHotScaler(object):
    def __init__(self, num_classes=-1, name=''):
        super().__init__()
        self.num_classes = num_classes
        self.name = name

    def fit(self, x):
        self.num_classes = torch.max(x)

    def transform(self, x):
        xt = torch.tensor(np.array(x, dtype=np.int64))
        oh = F.one_hot(xt, num_classes=self.num_classes)
        if len(oh.shape) == 4:
            oh = oh.permute(0, 3, 1, 2)
            dims = ('batch', f'{self.name}cat', 'y', 'x')
        elif len(oh.shape) == 3:
            oh = oh.permute(2, 0, 1)
            dims = (f'{self.name}cat', 'y', 'x')
        elif len(oh.shape) == 2:
            dims = ('sample', f'{self.name}cat')

        if isinstance(x, xr.DataArray):
            x = xr.DataArray(oh.numpy(), dims=dims)
            return x
        return oh.numpy()

    def inverse_transform(self, x):
        """Probably can be implemented with torch.argmax"""
        raise NotImplementedError()

class MinMaxScaler(BaseScaler):
    """
    The MinMaxScaler simply normalizes the data based on
    minimum and maximum values into a given feature range.
    """

    def __init__(self, x_min=None, x_max=None, feature_range=(0,1), **kwargs):
        self.eps = 1e-6
        self.x_min = x_min
        self.x_max = x_max

    def fit(self, x):
        self.x_min = x.min()
        self.x_max = x.max()

    def transform(self, x):
        x_scaled = (x - self.x_min) / (self.x_max - self.x_min)
        return x_scaled

    def inverse_transform(self, y):
        x = ((self.x_max - self.x_min) * y) + self.x_min
        return x#.view(orig_shape)

    def tensor_inverse_transform(self, y):
        xmax = utils.match_dims(torch.tensor(self.x_max.values), y).to(y.dtype).to(y.device)
        xmin = utils.match_dims(torch.tensor(self.x_min.values), y).to(y.dtype).to(y.device)
        return (xmax-xmin) * y + xmin


class StandardScaler(BaseScaler):
    """
    The StandardScaler standardizes data by subtracting the
    mean and dividing by the standard deviation of the data
    """

    def __init__(self, mean=None, std=None, **kwargs):
        self.eps = 1e-6
        self.mean = mean
        self.std = std

    def fit(self, x, unbiased=False):
        self.mean = x.mean()
        self.std = x.std(unbiased=unbiased)

    def transform(self, x):
        y = (x - self.mean) / (self.std + self.eps)
        return y

    def inverse_transform(self, y):
        x = y * (self.std + self.eps) + self.mean
        return x


def create_scalers_from_yaml(file):
    with open(file, 'r') as f:
        lookup = yaml.load(f, Loader=yaml.FullLoader)
    scalers = {}
    for k, v in lookup.items():
        if v['type'] == 'StandardScaler':
            scalers[k] = StandardScaler(float(v['mean']), float(v['std']))
        elif v['type'] == 'MinMaxScaler':
            scalers[k] = MinMaxScaler(float(v['min']), float(v['max']))
            print("Need to turn minmax scalers back on!")
    return scalers


DEFAULT_SCALERS = create_scalers_from_yaml(DEFAULT_SCALER_PATH)
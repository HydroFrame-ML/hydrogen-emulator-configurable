import os
import numpy as np
import scipy.stats
import torch
import xarray as xr
import pickle
import dill
import torch.nn.functional as F
from . import utils


def from_defaults(defaults):
    return {k: cls(**kwargs) for k, (cls, kwargs) in defaults.items()}


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
    return from_defaults(scaler_values)


class BaseScaler:
    """ Implements base methods that all scalers should inherit """

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

'''
class OneHotScaler(BaseScaler):

    def __init__(self, num_classes=-1):
        super().__init__()
        self.num_classes = num_classes

    def fit(self, x):
        self.num_classes = torch.max(x)

    def transform(self, x):
        x = torch.tensor(np.array(x, dtype=np.int64))
        oh = F.one_hot(x, num_classes=self.num_classes)
        oh = oh.permute(tuple(np.roll(np.arange(oh.ndim), 1)))
        return oh.numpy()

    def inverse_transform(self, x):
        """Probably can be implemented with torch.argmax"""
        raise NotImplementedError()
'''

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

class MedianAndQuantileScaler(BaseScaler):

    def __init__(self, median_file, quantile_file, q_range=[0.9, 0.1], **kwargs):
        self.median = xr.open_dataarray(median_file)
        self.high = xr.open_dataarray(quantile_file).sel(quantile=q_range[0])
        self.low = xr.open_dataarray(quantile_file).sel(quantile=q_range[1])

    def fit(self, x):
        pass

    def transform(self, x, eps=1e-12):
        sel_med = self.median.sel(**x.coords)
        sel_low = self.low.sel(**x.coords)
        sel_high = self.high.sel(**x.coords)
        return (x - sel_med) / (sel_high - sel_low + eps)

    def inverse_transform(self, x, eps=1e-12):
        sel_med = self.median.sel(**x.coords)
        sel_low = self.low.sel(**x.coords)
        sel_high = self.high.sel(**x.coords)
        return (x - sel_med) / (sel_high - sel_low + eps)


class ScalarFunctionScaler(BaseScaler):

    def __init__(self, f, f_inverse):
        self.f = f
        self.f_inverse = f_inverse

    def fit(self, x):
        pass

    def transform(self, x):
        return self.f(x)

    def inverse_transform(self, xhat):
        return self.f_inverse(xhat)


class BoxCoxScaler(BaseScaler):
    """
    The BoxCoxScaler provides a way to scale skewed distributions
    towards a more normal distribution.
    """

    def __init__(self, lmbda=None, shift=None, **kwargs):
        self.lmbda = lmbda
        self.shift = shift
        self.eps = 1e-6

    def fit(self, x):
        x = x.view(-1)
        self.shift = x.min() - self.eps
        self.lmbda = self._estimate_lmbda(x - self.shift)

    def transform(self, x):
        orig_shape = x.shape
        x = x.view(-1)
        if self.lmbda == 0:
            y = torch.log(x - self.shift)
        else:
            y = (torch.pow(x, self.lmbda) - 1) / self.lmbda
        return y.view(orig_shape)

    def inverse_transform(self, y):
        orig_shape = y.shape
        y = y.view(-1)
        if self.lmbda == 0:
            x = torch.exp(y)
        else:
            x = torch.pow(self.lmbda * y + 1, 1/self.lmbda)
        return x.view(orig_shape)

    def _estimate_lmbda(self, x):
        return scipy.stats.boxcox_normmax(x)


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


class CompositeScaler(BaseScaler):
    """
    CompositeScalers can be used to apply multiple transforms
    """

    def __init__(self, scaler_values=[]):
        self.scalers = scaler_values

    def fit(self, x):
        for s in self.scalers:
            s.fit(x)
            # Need to transform the current input
            # for the next scaler's fit
            x = s.transform(x)

    def transform(self, x):
        for s in self.scalers:
            x = s.transform(x)
        return x

    def inverse_transform(self, y):
        for s in self.scalers:
            y = s.inverse_transform(y)
        return y

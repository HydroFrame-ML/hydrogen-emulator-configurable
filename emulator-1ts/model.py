import torch
import torch.nn.functional as F
from torch import nn
from scalers import DEFAULT_SCALERS

def get_model(
    model_name,
    model_def
):
    if model_name == 'resnet':
        return ResNet(**model_def)
    else:
        raise ValueError(f'Model {model_name} not found')

class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation()
        self.padding = int(kernel_size / 2)
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode='reflect'
        )

    def forward(self, x):
        return self.activation(self.conv(x))


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        activation,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.activation = activation()
        self.depthwise_conv = nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size / 2),
            padding_mode='reflect',
            groups=self.in_channels
        )
        self.layer_norm = LayerNorm2D(self.hidden_channels)
        self.pointwise_conv = nn.Conv2d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1
        )

    def forward(self, x):
        identity = x
        out = self.depthwise_conv(x)
        out = self.layer_norm(out)
        out = self.pointwise_conv(out)
        out = self.activation(out + identity)
        return out



class ResNet(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        kernel_size=5,
        depth=1,
        activation=nn.GELU,
        scalers=DEFAULT_SCALERS,
        pressure_names=None,
        evaptrans_names=None,
        param_names=None
    ):
        super().__init__()
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.activation = activation
        self.scalers = scalers
        self.pressure_names = pressure_names
        self.evaptrans_names = evaptrans_names
        self.param_names = param_names

        self.layers = [
            ConvBlock(
                self.input_channels,
                self.hidden_dim,
                self.kernel_size,
                self.activation,
            )
        ]
        for i in range(self.depth):
            self.layers.append(
                ResidualBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    self.kernel_size,
                    self.activation,
                )
            )
        self.layers.append(
            ConvBlock(
                self.hidden_dim,
                self.output_channels,
                1,
                nn.Identity
            )
        )
        self.layers = nn.ModuleList(self.layers)

    #@torch.jit.script_method
    def scale_pressure(self, x):
        # Dims are (batch, z, y, x)
        for i in range(x.shape[1]):
            mu = self.scalers[f'press_diff_{i}'].mean
            sigma = self.scalers[f'press_diff_{i}'].std
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
    
    #@torch.jit.script_method
    def unscale_pressure(self, x):
        # Dims are (batch, z, y, x)
        for i in range(x.shape[1]):
            mu = self.scalers[f'press_diff_{i}'].mean
            sigma = self.scalers[f'press_diff_{i}'].std
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu
    
    #@torch.jit.script_method
    def scale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name].mean
            sigma = self.scalers[name].std
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
    
    #@torch.jit.script_method
    def unscale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name].mean
            sigma = self.scalers[name].std
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu
    
    #@torch.jit.script_method
    def scale_statics(self, x):
        for i, name in enumerate(self.param_names):
            mu = self.scalers[name].mean
            sigma = self.scalers[name].std
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma

    
    #@torch.jit.script_method
    def unscale_statics(self, x):
        for i, name in enumerate(self.param_names):
            mu = self.scalers[name]['mean']
            sigma = self.scalers[name]['std']
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    #@torch.jit.script_method
    def forward(self, pressure, evaptrans, statics):
        # Scale the data
        self.scale_pressure(pressure)
        self.scale_evaptrans(evaptrans)
        self.scale_statics(statics)

        # Concatenate the data
        x = torch.cat([pressure, evaptrans, statics], dim=1)

        for l in self.layers:
            x = l(x)

        # Unscale the data
        self.unscale_pressure(pressure)
        return x

    def save_model(self, output_path):
       torch.jit.save(self, output_path)


import torch
import torch.nn.functional as F
from torch import nn
from scalers import DEFAULT_SCALERS
from typing import Dict

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
        print(f">>>>>>>>>>>>>>>>>>>>> f.layer_norm x dtype: {x.dtype}, weight dtype: {self.weight.dtype}, bias dtype: {self.bias.dtype}")
        self.weight = self.weight.to(torch.float64)
        self.bias = self.bias.to(torch.float64)
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
        ).double()

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
        ).double()
        self.layer_norm = LayerNorm2D(self.hidden_channels)
        self.pointwise_conv = nn.Conv2d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1
        ).double()

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
        param_names=None,
        n_evaptrans=None,
        parameter_list=None,
        param_nlayer=None
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
        self.n_evaptrans = n_evaptrans
        self.param_names = param_names
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer
        
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

    @torch.jit.export
    def get_parflow_pressure(self, pressure):
        pressure = pressure.unsqueeze(0)
        self.scale_pressure(pressure)
        return pressure
        
    @torch.jit.export
    def scale_pressure(self, x):
        # Dims are (batch, z, y, x)
        for i in range(x.shape[1]):
            mu = self.scalers[f'press_diff_{i}'][0]
            sigma = self.scalers[f'press_diff_{i}'][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
            
    @torch.jit.export
    def unscale_pressure(self, x):
        # Dims are (batch, z, y, x)
        for i in range(x.shape[1]):
            mu = self.scalers[f'press_diff_{i}'][0]
            sigma = self.scalers[f'press_diff_{i}'][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    @torch.jit.export
    def get_predicted_pressure(self, x):
        self.unscale_pressure(x)
        return x.squeeze()

    @torch.jit.export
    def get_parflow_evaptrans(self, evaptrans):
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        #Grab the top n_lay layers
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        evaptrans = evaptrans.unsqueeze(0)
        self.scale_evaptrans(evaptrans)
        return evaptrans
    
    @torch.jit.export
    def scale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
        
    @torch.jit.export
    def unscale_evaptrans(self, x):
        # Dims are (batch, z, y, x)
        for i, name in enumerate(self.evaptrans_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    @torch.jit.export
    def get_parflow_statics(self, statics:Dict[str, torch.Tensor]):
        parameter_data = []
        print("Parameter list: ", self.parameter_list)
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            print("Inside get_parflow_statics, ", parameter)
            param_temp = statics[parameter]
            if param_temp.shape[0] > 1:
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]
            print(param_temp.shape)
            parameter_data.append(param_temp)

        # Concatenate the parameter data together
        # End result is a dims of (n_parameters, y, x)
        parameter_data = torch.cat(parameter_data, dim=0)
        parameter_data = parameter_data.unsqueeze(0)
        print("statics shape: ", parameter_data.shape)
        self.scale_statics(parameter_data)
        print("statics shape: ", parameter_data.shape)
        return parameter_data
            
    @torch.jit.export
    def scale_statics(self, x):
        print("param names: ", self.param_names)
        for i, name in enumerate(self.param_names):
            print("Scale ", i, name)
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = (x[:, i, :, :] - mu) / sigma
    
    @torch.jit.export
    def unscale_statics(self, x):
        for i, name in enumerate(self.param_names):
            mu = self.scalers[name][0]
            sigma = self.scalers[name][1]
            x[:, i, :, :] = x[:, i, :, :] * sigma + mu

    def forward(self, pressure, evaptrans, statics):
        # Concatenate the data
        x = torch.cat([pressure, evaptrans, statics], dim=1)

        for l in self.layers:
            x = l(x)

        return x



from hydroml.loss import MWSE, DWSE
from functools import partial
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from typing import Optional, Union, List, Mapping

from tqdm.autonotebook import tqdm

import numpy as np
import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import model_builder

@model_builder.register_layer('ActionSTLSTMCell')
class ActionSTLSTMCell(nn.Module):
    def __init__(self, in_channel, action_channel, num_hidden, filter_size, stride):
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Conv2d(in_channel,     num_hidden * 7, filter_size, stride, self.padding)
        self.conv_a = nn.Conv2d(action_channel, num_hidden * 4, filter_size, stride, self.padding)
        self.conv_h = nn.Conv2d(num_hidden,     num_hidden * 4, filter_size, stride, self.padding)
        self.conv_m = nn.Conv2d(num_hidden,     num_hidden * 3, filter_size, stride, self.padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden,     filter_size, stride, self.padding)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, a_t, h_t, c_t, m_t):
        x_conv = self.conv_x(x_t)
        a_conv = self.conv_a(a_t)
        h_conv = self.conv_h(h_t)
        m_conv = self.conv_m(m_t)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_conv, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_conv * a_conv, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_conv, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + self._forget_bias)
        g_tp = torch.tanh(g_xp + g_m)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c
        delta_m = i_tp * g_tp
        m_new = f_tp * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new, delta_c, delta_m


@model_builder.register_model('ForcedSTRNN')
class ForcedSTRNN(pl.LightningModule):
    def __init__(
        self,
        num_layers,
        num_hidden,
        img_channel,
        act_channel,
        init_cond_channel,
        static_channel,
        out_channel,
        img_width,
        device,
        filter_size=5,
        stride=1,
    ):
        super().__init__()

        self.input_channel = img_channel
        self.action_channel = act_channel
        self.init_cond_channel = init_cond_channel
        self.static_channel = static_channel
        self.frame_channel = img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.device = device
        self.out_channel = out_channel
        self.decouple_loss = None
        cell_list = []

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(ActionSTLSTMCell(
                in_channel, act_channel, num_hidden[i], filter_size, stride,
            ))
        self.cell_list = nn.ModuleList(cell_list)
        # Set up the output conv layer, which
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.out_channel,
            kernel_size=1,
            bias=False
        )
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden,
            adapter_num_hidden,
            kernel_size=1,
            bias=False
        )

        self.memory_encoder = nn.Conv2d(
            self.init_cond_channel, num_hidden[0], kernel_size=1, bias=False
        )
        # Should we encode different layers separately?
        self.cell_encoder = nn.Conv2d(
            self.static_channel, num_hidden[0], kernel_size=1, bias=False
        )

    def update_state(self, state):
        out_shape = (state.shape[0], state.shape[1], -1)
        return F.normalize(self.adapter(state).view(out_shape), dim=2)

    def calc_decouple_loss(self, c, m):
        return torch.mean(torch.abs(torch.cosine_similarity(c, m, dim=2)))

    def forward(self, forcings, init_cond, static_inputs):
        # Input shape:
        #   (batch, length, channel, height, width)
        batch, timesteps, channels, height, width = forcings.shape

        # Initialize list of states
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        # Initialize memory state and first layer cell state
        # Note: static_inputs and init_cond should have length=1
        # TODO: Fix this requirement
        memory = self.memory_encoder(init_cond[:, 0])
        # TODO: Should we encode all layer cell states?
        c_t[0] = self.cell_encoder(static_inputs[:, 0])

        # First input is the initial condition
        x = init_cond[:, 0]
        for t in range(timesteps):
            a = forcings[:, t]
            h_t[0], c_t[0], memory, dc, dm = self.cell_list[0](x, a, h_t[0], c_t[0], memory)
            delta_c_list[0] = self.update_state(dc)
            delta_m_list[0] = self.update_state(dm)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, dc, dm = self.cell_list[i](h_t[i - 1], a, h_t[i], c_t[i], memory)
                delta_c_list[i] = self.update_state(dc)
                delta_m_list[i] = self.update_state(dm)

            x = self.conv_last(h_t[-1]) + x
            next_frames.append(x)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(self.calc_decouple_loss(delta_c_list[i], delta_m_list[i]))

        self.decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # Stack to: [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=1)
        return next_frames

    def training_step(self, train_batch, train_batch_idx):
        forcing, state, params, target = train_batch
        y_hat = self(forcing, state, params).squeeze()
        loss = self.loss_fun(y_hat, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        forcing, state, params, target = val_batch
        y_hat = self(forcing, state, params).squeeze()
        y_hat = self(target).squeeze()
        loss = self.loss_fun(y_hat, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(
        self,
        opt=torch.optim.AdamW,
    ):
        optimizer = opt(self.model.parameters(), lr=0.001)
        #scheduler = ExponentialLR(optimizer, gamma=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=0.001,
                total_steps=self.steps_per_epoch * self.max_epochs)
        return [optimizer], [scheduler]

    def configure_loss(self, loss_fun=DWSE):
        self.loss_fun = loss_fun



@model_builder.register_model('BaseLSTM')
class BaseLSTM(pl.LightningModule):
    """
    A basic wrapper around the nn.LSTM module.
    This allows for some nice user configuration options
    for the HydroGEN project, and makes it compatible with
    pytorh-lightning for simpler training scripts.

    Parameters
    ----------
    in_vars:
        TODO: FIXME
    out_vars:
        TODO: FIXME
    hidden_dim:
        The dimension of the hidden layers
    n_layers:
        The number of LSTM layers
    dropout_prob:
        Probability for dropout between LSTM layers
    sequence_length:
        The length of the `sequence_length` dimension
        in an output sample
    """

    def __init__(
        self,
        in_vars,
        out_vars,
        hidden_dim,
        nlayers,
        dropout_prob=0.0,
        sequence_length=1
    ):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        n_input_features = len(self.in_vars)
        n_output_features = len(self.out_vars)
        self.lstm = nn.LSTM(
            n_input_features,
            hidden_dim,
            nlayers,
            dropout=dropout_prob,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_output_features)
        self.sequence_length = sequence_length

    def forward(self, x):
        """Apply the model for a given sample"""
        out, (hn, cn) = self.lstm(x)
        out = out[:, -self.sequence_length:, :]
        out = self.fc(out)
        return out

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.fc.parameters())

    def configure_optimizers(self, opt=torch.optim.AdamW, **kwargs):
        optimizer = opt(self.parameters(), **kwargs)
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        y = y[:, -self.sequence_length:, :].squeeze()
        y_hat = y_hat.squeeze()

        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        y = y[:, -self.sequence_length:, :].squeeze()
        y_hat = y_hat.squeeze()

        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)



@model_builder.register_emulator('MultiLSTMModel')
class MultiLSTMModel(pl.LightningModule):
    """
    A wrapper around the LSTMModel, which
    represents each output variable with it's own
    LSTM with it's own trainable weights/biases.
    """
    def __init__(
        self,
        forcing_vars,
        surface_parameters,
        subsurface_parameters,
        state_vars,
        out_vars,
        sequence_length,
        layer_model=BaseLSTM,
        layer_model_kwargs={},
    ):
        super().__init__()
        if 'vegtype' in surface_parameters:
            extra_vars = [f'vegtype_{i}' for i in range(1, 18)]
            surface_parameters = surface_parameters + extra_vars
        if 'vegetation_type' in surface_parameters:
            extra_vars = [f'vegetation_type_{i}' for i in range(1, 18)]
            surface_parameters = surface_parameters + extra_vars
        self.in_vars = (forcing_vars + surface_parameters
                        + subsurface_parameters + state_vars)
        self.out_vars = out_vars
        self.sequence_length = sequence_length
        n_input_features = len(self.in_vars)
        n_output_features = len(self.out_vars)
        self.lstmlist = nn.ModuleList([
            layer_model(
                in_vars=self.in_vars,
                out_vars=['_'],
                sequence_length=self.sequence_length,
                **layer_model_kwargs,
            ) for _ in range(n_output_features)
        ])

    def forward(self, x):
        """Apply the model for a given sample"""
        yhat = []
        for lstm in self.lstmlist:
            out = lstm(x).squeeze()
            yhat.append(out)
        yhat = torch.stack(yhat, axis=-1)
        yhat = yhat[..., -self.sequence_length:]
        return yhat

    def configure_optimizers(
        self,
        opt=torch.optim.AdamW,
    ):
        optimizer = opt(self.parameters(), lr=0.0001)
        #scheduler = ExponentialLR(optimizer, gamma=0.9)
        #scheduler = OneCycleLR(optimizer, max_lr=0.1,
        #        total_steps=self.steps_per_epoch * self.max_epochs)
        #return [optimizer], [scheduler]
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        y = y[:, -self.sequence_length:, :].squeeze()
        y_hat = y_hat.squeeze()

        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        y = y[:, -self.sequence_length:, :].squeeze()
        y_hat = y_hat.squeeze()

        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)



@model_builder.register_layer('DoubleConv')
class DoubleConv(nn.Module):
    """(convolution => SeLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        activation=nn.SELU,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.activation = activation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            self.activation(),#(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            self.activation(),#(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


@model_builder.register_layer('Down')
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation=nn.SELU):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


@model_builder.register_layer('Up')
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        activation=nn.SELU
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
                activation
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                activation=activation
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@model_builder.register_layer('OutConv')
class OutConv(nn.Module):
    """Simple wrapper for an ouput convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@model_builder.register_model('UNet')
class UNet(pl.LightningModule):
    """
    A basic UNet architecture
    """

    def __init__(
        self,
        in_vars=[''],
        out_vars=[''],
        activation=nn.Mish,
        bilinear=True,
        base_channels=8
    ):
        super().__init__()
        self.in_channels = len(in_vars)
        self.out_channels = len(out_vars)
        self.bilinear = bilinear
        c = base_channels

        self.inc = DoubleConv(self.in_channels, c, activation=activation)
        self.down1 = Down(c, 2*c, activation=activation)
        self.down2 = Down(2*c, 2*2*c, activation=activation)
        self.down3 = Down(2*2*c, 2*2*2*c, activation=activation)
        factor = 2 if bilinear else 1
        self.down4 = Down(2*2*2*c, 2*2*2*2*c// factor, activation=activation)
        self.up1 = Up(2*2*2*2*c, 2*2*2*c// factor, bilinear, activation=activation)
        self.up2 = Up(2*2*2*c, 2*2*c// factor, bilinear, activation=activation)
        self.up3 = Up(2*2*c, 2*c// factor, bilinear, activation=activation)
        self.up4 = Up(2*c, c, bilinear, activation=activation)
        self.outc = OutConv(c, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        z = self.outc(x)
        return z

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self, opt=torch.optim.AdamW, **kwargs):
        optimizer = opt(self.parameters(), **kwargs)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun


@model_builder.register_layer('ConvBlock')
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
            padding=self.padding
        )

    def forward(self, x):
        return self.activation(self.conv(x))


@model_builder.register_layer('ResidualBlock')
class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        activation,
        depth,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation()
        self.depth = depth
        self.conv_block = [
            ConvBlock(
                in_channels,
                hidden_channels,
                kernel_size,
                activation
            )
        ]
        for i in range(1, self.depth-1):
            self.conv_block.append(
                ConvBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    activation
                )
            )
        self.conv_block.append(
            ConvBlock(
                hidden_channels,
                out_channels,
                kernel_size,
                nn.Identity
            )
        )
        self.conv_block = nn.ModuleList(self.conv_block)

    def forward(self, x):
        identity = x
        for l in self.conv_block:
            x = l(x)
        x = x + identity
        out = self.activation(x)
        return out


@model_builder.register_model('BasicConvNet')
class BasicConvNet(pl.LightningModule):

    def __init__(
        self,
        in_vars=[''],
        out_vars=[''],
        hidden_dim=64,
        kernel_size=5,
        depth=3,
        activation=nn.Mish,
    ):
        super().__init__()
        self.input_channels = len(in_vars)
        self.hidden_dim = hidden_dim
        self.output_channels = len(out_vars)
        self.kernel_size = kernel_size
        self.depth = depth
        self.activation = activation

        self.layers = [
            ConvBlock(
                self.input_channels,
                self.hidden_dim,
                self.kernel_size,
                self.activation
            )
        ]
        for i in range(1, self.depth-1):
            self.layers.append(
                ConvBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.kernel_size,
                    self.activation
                )
            )
        self.layers.append(
            ConvBlock(
                self.hidden_dim,
                self.output_channels,
                self.kernel_size,
                nn.Identity
            )
        )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def configure_optimizers(self, opt=torch.optim.Adam, **kwargs):
        optimizer = opt(self.parameters(), **kwargs)
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss


@model_builder.register_model('BasicResNet')
class BasicResNet(pl.LightningModule):

    def __init__(
        self,
        in_vars=[''],
        out_vars=[''],
        hidden_dim=64,
        kernel_size=5,
        res_block_depth=1,
        total_depth=1,
        activation=nn.Mish,
    ):
        super().__init__()
        self.input_channels = len(in_vars)
        self.hidden_dim = hidden_dim
        self.output_channels = len(out_vars)
        self.kernel_size = kernel_size
        self.depth = total_depth
        self.res_block_depth = res_block_depth
        self.activation = activation

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
                    self.res_block_depth,
                )
            )
        self.layers.append(
            ConvBlock(
                self.hidden_dim,
                self.output_channels,
                self.kernel_size,
                nn.Identity
            )
        )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        identity = x
        for l in self.layers:
            x = l(x)
        return x

    def configure_optimizers(self, opt=torch.optim.Adam, **kwargs):
        optimizer = opt(self.parameters(), **kwargs)
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss


@model_builder.register_emulator('MultiStepModel')
class MultiStepModel(pl.LightningModule):
    def __init__(
        self,
        forcing_vars,
        surface_parameters,
        subsurface_parameters,
        state_vars,
        out_vars,
        sequence_length=7,
        nlayers=5,
        layer_model=BasicConvNet,
        layer_model_kwargs={},
        stepping_index=-1,
        probability_of_true_inputs=0.0,
        corrector_mode=False,
        inject_noise=False,
        rng=np.random,
    ):
        super().__init__()
        self.nlayers = nlayers
        self.forcing_vars = forcing_vars
        self.pf_parameters = surface_parameters + self.nlayers * subsurface_parameters
        self.state_vars = state_vars * self.nlayers
        self.out_vars = out_vars * self.nlayers
        self.stepping_index = stepping_index
        self.corrector_mode = corrector_mode
        self.in_vars = self.forcing_vars+self.pf_parameters+self.state_vars
        print(len(self.in_vars), len(self.out_vars), self.corrector_mode)

        self.model = layer_model(
            self.in_vars,
            self.out_vars,
            **layer_model_kwargs
        )
        self.sequence_length = sequence_length
        self.probability_of_true_inputs = probability_of_true_inputs
        self.inject_noise = inject_noise
        self.rng = rng
        self.save_hyperparameters(
            'forcing_vars', 'surface_parameters', 'subsurface_parameters', 'state_vars',
            'sequence_length', 'probability_of_true_inputs', 'inject_noise'
        )

    def forward(self, x):
        # x has dims (batch, steps, features, height, width)
        # So now xx has dims (batch, features, height, width)
        xx = x[:, 0].clone()
        # y_hat_sub stores each timestep prediction
        y_hat_sub = []
        pred = self.model(xx)#.squeeze()
        #print(pred.shape, xx.shape)
        if self.corrector_mode == True:
            y_hat_sub.append(pred + xx[:, -self.stepping_index:, :, :])
        else:
            y_hat_sub.append(pred)
        noise_scale = 1e-6
        # Iterate through time
        for i in range(1, self.sequence_length):
            # Get the ith timestep
            xx = x[:, i].clone()
            # Now copy in the pressure prediction
            # from the previous timestep
            #replace=1
            xx[:, -self.stepping_index, :, :] = (
                y_hat_sub[-1].squeeze().detach().clone())
            pred = self.model(xx)#.squeeze()
            if self.corrector_mode == True:
                y_hat_sub.append(pred + y_hat_sub[-1])
            else:
                y_hat_sub.append(pred)
        # Stack everything together to get dims
        # (batch, steps, features, height, width)
        return torch.stack(y_hat_sub, axis=1)



    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(
        self,
        opt=torch.optim.AdamW,
    ):
        optimizer = opt(self.parameters(), lr=0.0003)
        return optimizer
        #scheduler = ExponentialLR(optimizer, gamma=0.9)
        #scheduler = OneCycleLR(optimizer, max_lr=0.01,
        #        total_steps=self.steps_per_epoch * self.max_epochs)
        #return [optimizer], [scheduler]

    def configure_loss(self, loss_fun=DWSE):
        #weights = torch.log(torch.arange(self.sequence_length) + 1.1)
        #weights = weights / torch.sum(weights)
        ##weights = weights.to(self.device)
        #loss_fun = partial(loss_fun, weights=weights)
        self.loss_fun = loss_fun


@model_builder.register_emulator('FiveLayerModel')
class FiveLayerModel(pl.LightningModule):
    """
    The FiveLayerModel is a set of interconnected models with some base
    implementation which can be configured separately. This class is meant
    to mimic the way in which a vertically discretized environmental model
    is set up, where each sub-model represents an individual layer in the
    discretization. These sub-models still represent a spatially-distributed
    area (conventionally described as the x,y coordinates). The inputs to each
    layer are partially defined by the adjacent layers. This implementation
    handles those details behind the scenes, along with some other features,
    all in the effort of keeping users from having to handle index juggling.

    Note that while we list out the dimensions of variability each input class
    can have, this model is inherently *not* temporal in nature and should be
    thought of as only seeing a single time slice at a time.

    Note that the term "layer" in this model documentation *always* refers to
    a single entry in a vertical discretization, rather than the concept of
    layers in deep-learning models.

    Note that all layers see the forcing data.

    Parameters
    ----------
    forcing_vars: List[str]
        The names of variables which represent meteorologic forcing which have
        variability in dimensions (time, y, x)
    surface_parameters: List[str]
        The names of variables which represent parameter values which
        only vary on the surface (top) layer, meaning variability in
        the (y, x) dimensions only.
    subsurface_parameters: List[str]
        The names of variables which represent subsurface parameter values
        which have 3d variability (z, y, x)
    state_vars: List[str]
        The names of variables which represent the subsurface state and have
        the full 4d variability (time, z, y, x).
    activation: nn.Module
        The activation function used for the `layer_model` (if applicable)
    layer_model: nn.Module
        The model structure that is used for each of the five discrete layers
        Note this is just the reference to the class and not an instance of it.
        Note this model must take in `grid_size`, `in_vars`, `out_vars`, and
        `activation` as constructor arguments.
    layer_model_kwargs: Mapping
        Arguments to pass into each sub-model for the discrete layers
    """

    def __init__(
        self,
        forcing_vars,
        surface_parameters,
        subsurface_parameters,
        state_vars,
        out_vars,
        layer_model=UNet,
        layer_model_kwargs={},
    ):
        super().__init__()
        self.forcing_vars = forcing_vars
        self.surface_parameters = surface_parameters
        self.subsurface_parameters = subsurface_parameters
        self.state_vars = state_vars
        self.out_vars = out_vars
        self.layer_model = layer_model
        self.layer_model_kwargs = layer_model_kwargs

        self.feature_indexers = self._gen_feature_indexers()
        #NOTE: We go bottom to top like parflow
        self.sub_models = nn.ModuleList([
            self._bot_layer_sub_model(),
            self._mid_layer_sub_model(has_forcing=False),
            self._mid_layer_sub_model(has_forcing=False),
            self._mid_layer_sub_model(has_forcing=False),
            self._top_layer_sub_model()
        ])

    def _top_layer_sub_model(self):
        """
        Creates the "top", or uppermost layer model. This model takes in
        forcing data, surface parameters, the top layer of subsurface
        parameters, and the state variables from the top two layers.
        """
        input_channels = (
                len(self.forcing_vars)
                + len(self.surface_parameters)
                + 2 * len(self.subsurface_parameters)
                + 2 * len(self.state_vars)
        )
        output_channels = len(self.out_vars)
        return self.layer_model(
                in_vars=['_' for i in range(input_channels)],
                out_vars=['_' for i in range(output_channels)],
                **self.layer_model_kwargs
        )

    def _mid_layer_sub_model(self, has_forcing=True):
        """
        Creates a "middle" layer model. This model takes in
        forcing data, a single layer of subsurface parameters,
        and the state variables from the three layers
        (one above, one below, and the local layer).
        """
        input_channels = (len(self.surface_parameters)
                          + 3 * len(self.subsurface_parameters)
                          + 3 * len(self.state_vars))
        if has_forcing:
            input_channels += len(self.forcing_vars)

        output_channels = len(self.out_vars)
        return self.layer_model(
                in_vars=['_' for i in range(input_channels)],
                out_vars=['_' for i in range(output_channels)],
                **self.layer_model_kwargs
        )

    def _bot_layer_sub_model(self):
        """
        Creates the "bottom" layer model. This model takes in
        forcing data, a single layer of subsurface parameters,
        and the state varaibles from the lowest two layes.
        """
        input_channels = (
                #len(self.forcing_vars)
                len(self.surface_parameters)
                + 2 * len(self.subsurface_parameters)
                + 2 * len(self.state_vars)
        )
        output_channels = len(self.out_vars)
        return self.layer_model(
                in_vars=['_' for i in range(input_channels)],
                out_vars=['_' for i in range(output_channels)],
                **self.layer_model_kwargs
        )

    def _gen_feature_indexers(self):
        """
        This method translates the featues to the indices needed
        to pull them out of an input sample. This is required because
        some of the `sub_models` receive the same input variables.
        """
        # Forcing index
        forc_idx = np.arange(len(self.forcing_vars))
        # Surface parameter index
        surf_p_idx = np.arange(
            len(self.forcing_vars),
            len(self.forcing_vars) + len(self.surface_parameters)
        )
        # Subsurface parameter index
        if not len(self.subsurface_parameters):
            sub_p_idx = np.array([np.nan for _ in range(5)])
        else:
            sub_p_idx = np.arange(
                len(self.forcing_vars) + len(self.surface_parameters),
                (len(self.forcing_vars)
                 + len(self.surface_parameters)
                 + 5 * len(self.subsurface_parameters))
            )
        # Subsurface state index
        state_idx = np.arange(
            (len(self.forcing_vars)
             + len(self.surface_parameters)
             + 5 * len(self.subsurface_parameters)),
            (len(self.forcing_vars)
             + len(self.surface_parameters)
             + 5 * len(self.subsurface_parameters)
             + 5 * len(self.state_vars))
        )
        ss_idx = lambda s: [
            (i*5)+s for i in range(len(self.subsurface_parameters))
        ]

        l = 0
        l0_subp_idx = np.hstack([
            np.array(sub_p_idx)[[(i*5)+l, (i*5)+l+1]]
            for i in range(len(self.subsurface_parameters))
        ])
        l = 1
        l1_subp_idx = np.hstack([
            np.array(sub_p_idx)[[(i*5)+l-1, (i*5)+l, (i*5)+l+1]]
            for i in range(len(self.subsurface_parameters))
        ])
        l = 2
        l2_subp_idx = np.hstack([
            np.array(sub_p_idx)[[(i*5)+l-1, (i*5)+l, (i*5)+l+1]]
            for i in range(len(self.subsurface_parameters))
        ])
        l = 3
        l3_subp_idx = np.hstack([
            np.array(sub_p_idx)[[(i*5)+l-1, (i*5)+l, (i*5)+l+1]]
            for i in range(len(self.subsurface_parameters))
        ])
        l = 4
        l4_subp_idx = np.hstack([
            np.array(sub_p_idx)[[(i*5)+l-1, (i*5)+l]]
            for i in range(len(self.subsurface_parameters))
        ])
        # Indices for 1st layer
        # then for the 2nd layer (and so on...)
        l0_idx = np.hstack([
            #forc_idx,
            surf_p_idx,
            l0_subp_idx,
            state_idx[[0, 1]]
        ])
        l1_idx = np.hstack([
            #forc_idx,
            surf_p_idx,
            l1_subp_idx,
            state_idx[[0, 1, 2]]
        ])
        l2_idx = np.hstack([
            #forc_idx,
            surf_p_idx,
            l2_subp_idx,
            state_idx[[1, 2, 3]]
        ])
        l3_idx = np.hstack([
            #forc_idx,
            surf_p_idx,
            l3_subp_idx,
            state_idx[[2, 3, 4]]
        ])
        l4_idx = np.hstack([
            forc_idx,
            surf_p_idx,
            l4_subp_idx,
            state_idx[[3, 4]]
        ])
        feature_indexers = [l0_idx, l1_idx, l2_idx, l3_idx, l4_idx]
        feature_indexers = [x[~np.isnan(x)] for x in feature_indexers]
        return feature_indexers

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def forward(self, x):
        """
        Run the predict step. The data in this case will have
        the format (channel, x, y) where channel will be indexed as

         - 0:n_frc -> forcing data
         - n_frc:n_par -> parameters
         - n_par:n_lyr -> layers
        """
        all_y = []
        for fi, sub_model in zip(self.feature_indexers, self.sub_models):
            pred = sub_model(x[:, fi])
            all_y.append(pred.squeeze(dim=1))
        y_hat = torch.stack(all_y, axis=1)
        return y_hat

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self, opt=torch.optim.AdamW, **kwargs):
        optimizer = opt(self.sub_models.parameters(), **kwargs)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]


@model_builder.register_emulator('MultiStepMultiLayerModel')
class MultiStepMultiLayerModel(pl.LightningModule):
    """
    The MultiStepMultiLayerModel wraps around the FiveLayerModel but adds
    time dependence into the simulation. This is done explicitly in the
    `forward` method.

    Parameters
    ----------
    forcing_vars: List[str]
        The names of variables which represent meteorologic forcing which have
        variability in dimensions (time, y, x)
    surface_parameters: List[str]
        The names of variables which represent parameter values which
        only vary on the surface (top) layer, meaning variability in
        the (y, x) dimensions only.
    subsurface_parameters: List[str]
        The names of variables which represent subsurface parameter values
        which have 3d variability (z, y, x)
    state_vars: List[str]
        The names of variables which represent the subsurface state and have
        the full 4d variability (time, z, y, x).
    sequence_length: int
        The number of timesteps which are simulated
    activation: nn.Module
        The activation function used for the `layer_model` (if applicable)
    layer_model: nn.Module
        The model structure that is used for each of the five discrete layers
        Note this is just the reference to the class and not an instance of it.
        Note this model must take in `in_vars`, `out_vars`, and
        `activation` as constructor arguments.
    layer_model_kwargs: dict
        Arguments to pass into each sub-model for the discrete layers
    probability_of_true_inputs: float
        The probability to use to inject the true state value rather than
        the one propagated forward in time during the time loop of the
        `forward` method. When using this model in inference mode this needs
        to be set to 0.0 to ensure that results are valid.
    inject_noise: bool
        Whether to inject noise into the inputs for the time looping. This
        has been shown to improve stability during training as well for
        unrolling the time loop over longer numbers of timesteps. If set to
        `True` then some small random noise is added to inputs.
    rng: callable
        The function to use as the random number generator if `inject_noise` is
        set to `True`
    """

    def __init__(
        self,
        forcing_vars,
        surface_parameters,
        subsurface_parameters,
        state_vars,
        out_vars,
        sequence_length=7,
        layer_model=UNet,
        layer_model_kwargs={},
        probability_of_true_inputs=0.0,
        inject_noise=True,
        corrector_mode=True,
        rng=np.random,
    ):
        super().__init__()
        self.save_hyperparameters(
            'forcing_vars', 'surface_parameters',
            'subsurface_parameters', 'state_vars', 'sequence_length',
            'probability_of_true_inputs', 'inject_noise'
        )
        self.model = FiveLayerModel(
            forcing_vars,
            surface_parameters,
            subsurface_parameters,
            state_vars,
            out_vars,
            layer_model=layer_model,
            layer_model_kwargs=layer_model_kwargs
        )
        self.corrector_mode = corrector_mode
        self.sequence_length = sequence_length
        self.probability_of_true_inputs = probability_of_true_inputs
        self.inject_noise = inject_noise
        print(self.corrector_mode, self.sequence_length, self.probability_of_true_inputs)
        self.rng = rng

    def forward(self, x):
        # x has dims (batch, steps, features, height, width)
        # So now xx has dims (batch, features, height, width)
        xx = x[:, 0].clone()
        # y_hat_sub stores each timestep prediction
        y_hat_sub = []
        pred = self.model(xx)#.squeeze()
        #print(pred.shape, xx.shape)
        if self.corrector_mode == True:
            y_hat_sub.append(pred + xx[:, -5:, :, :])
        else:
            y_hat_sub.append(pred)
        noise_scale = 1e-6
        # Iterate through time
        for i in range(1, self.sequence_length):
            # Get the ith timestep
            xx = x[:, i].clone()
            # Now copy in the pressure prediction
            # from the previous timestep
            #replace=1
            #if self.rng.random() > self.probability_of_true_inputs:
            xx[:, -5:, :, :] = y_hat_sub[-1].squeeze().detach().clone()
            #if self.inject_noise:
            #    noise = torch.from_numpy(
            #        self.rng.random(size=xx.shape).astype(np.float32)
            #    ).to(self.device)
            #    xx = xx + noise_scale * noise
            pred = self.model(xx)#.squeeze()
            if self.corrector_mode == True:
                y_hat_sub.append(pred + xx[:, -5:, : , :])
            else:
                y_hat_sub.append(pred)
        # Stack everything together to get dims
        # (batch, steps, features, height, width)
        return torch.stack(y_hat_sub, axis=1)

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self(x).squeeze()
        #loss = self.loss_fun(y_hat[..., 2:-2, 2:-2], y[..., 2:-2, 2:-2])
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self(x).squeeze()
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(
        self,
        opt=torch.optim.AdamW,
    ):
        optimizer = opt(self.model.parameters(), lr=0.0001)
        #scheduler = ExponentialLR(optimizer, gamma=0.9)
        #scheduler = OneCycleLR(optimizer, max_lr=0.01,
        #        total_steps=self.steps_per_epoch * self.max_epochs)
        #return [optimizer], [scheduler]
        return optimizer

    def configure_loss(self, loss_fun=DWSE):
        weights = torch.log(torch.arange(self.sequence_length) + 1.1)
        weights = weights / torch.sum(weights)
        #weights = weights.to(self.device)
        #loss_fun = partial(loss_fun, weights=weights)
        self.loss_fun = loss_fun



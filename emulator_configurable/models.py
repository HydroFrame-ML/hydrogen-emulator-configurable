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

class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


@model_builder.register_layer('ActionSTLSTMCell')
class ActionSTLSTMCell(nn.Module):
    def __init__(self, in_channel, action_channel, num_hidden, filter_size, stride):
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 7)
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(action_channel, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 3)
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden,     filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden)
        )
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


@model_builder.register_emulator('ForcedSTRNN')
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
            self.init_cond_channel, num_hidden[0], kernel_size=1, bias=True
        )
        self.cell_encoder = nn.Conv2d(
            self.static_channel, sum(num_hidden), kernel_size=1, bias=True
        )

    def update_state(self, state):
        out_shape = (state.shape[0], state.shape[1], -1)
        return F.normalize(self.adapter(state).view(out_shape), dim=2)

    def calc_decouple_loss(self, c, m):
        return torch.mean(torch.abs(torch.cosine_similarity(c, m, dim=2)))

    def forward(self, forcings, init_cond, static_inputs):
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
        memory = self.memory_encoder(init_cond[:, 0])
        c_t = list(torch.split(
            self.cell_encoder(static_inputs[:, 0]),
            self.num_hidden, dim=1
        ))

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
                decouple_loss.append(self.calc_decouple_loss(delta_c_list[i], delta_m_list[i]))

            x = self.conv_last(h_t[-1]) + x
            next_frames.append(x)

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
        loss = self.loss_fun(y_hat, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(
        self,
        opt=torch.optim.AdamW,
    ):
        optimizer = opt(self.parameters(), lr=self.learning_rate, betas=[0.8, 0.95])
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        def _inner_loss(yhat, y):
            return loss_fun(yhat, y) + self.decouple_loss
        self.loss_fun = _inner_loss



@model_builder.register_model('BaseLSTM')
class BaseLSTM(pl.LightningModule):
    """
    A basic wrapper around the nn.LSTM module.
    This allows for some nice user configuration options
    for the HydroGEN project, and makes it compatible with
    pytorh-lightning for simpler training scripts.

    Parameters
    ----------
    in_features : int
        The number of input features
    out_features : int
        The number of output features
    hidden_dim : int
        The number of hidden features
    nlayers : int
        The number of layers
    dropout_prob : float
        The dropout probability
    sequence_length : int
        The number of timesteps to use for the LSTM
    """

    def __init__(
        self,
        in_features,
        out_features,
        hidden_dim,
        nlayers,
        dropout_prob=0.0,
        sequence_length=1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            self.in_features,
            self.hidden_dim,
            nlayers,
            dropout=dropout_prob,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, self.out_features)

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


@model_builder.register_layer('DoubleConv')
class DoubleConv(nn.Module):
    """(convolution => GELU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        activation=nn.GELU,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.activation = activation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            self.activation(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            self.activation(),
        )

    def forward(self, x):
        return self.double_conv(x)


@model_builder.register_layer('Down')
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation=nn.GELU):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
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
        activation=nn.GELU
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
class UNet(nn.Module):
    """
    A basic UNet architecture
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        activation=nn.GELU,
        bilinear=True,
        base_channels=8
    ):
        super().__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
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
            padding=self.padding,
            padding_mode='reflect'
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


@model_builder.register_model('BasicResNet')
class BasicResNet(pl.LightningModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        kernel_size=5,
        depth=1,
        activation=nn.GELU,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.depth = depth
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

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


@model_builder.register_emulator('MultiStepModel')
class MultiStepModel(pl.LightningModule):
    def __init__(
        self,
        in_channel,
        out_channel,
        layer_model=UNet,
        layer_model_kwargs={},
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.model = layer_model(
            self.in_channel,
            self.out_channel,
            **layer_model_kwargs
        )

    def forward(self, forcings, init_cond, static_inputs):
        batch, timesteps, channels, height, width = forcings.shape

        next_frames = []

        # NOTE: init_cond and static_inputs have length=1 on time
        x = init_cond[:, 0]
        s = static_inputs[:, 0]
        for t in range(timesteps):
            f_t = forcings[:, t]
            inp = torch.cat([f_t, s, x], dim=1)
            out = self.model(inp)
            x = out + x
            next_frames.append(x)
        next_frames = torch.stack(next_frames, dim=1)
        return next_frames

    def training_step(self, train_batch, train_batch_idx):
        forcing, state, params, target = train_batch
        y_hat = self(forcing, state, params).squeeze()
        loss = self.loss_fun(y_hat, target)
        if torch.isnan(loss):
            print(torch.isnan(target).sum(), torch.isnan(y_hat).sum())
            raise ValueError('Loss went nan')

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        forcing, state, params, target = val_batch
        y_hat = self(forcing, state, params).squeeze()
        loss = self.loss_fun(y_hat, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(
        self,
        opt=torch.optim.AdamW,
    ):
        optimizer = opt(self.parameters(), lr=self.learning_rate, betas=[0.8, 0.95])
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

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
from .process_heads import (
    SaturationHead,
    WaterTableDepthHead,
    OverlandFlowHead
)
from .scalers import DEFAULT_SCALERS



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
        self.use_decouple_loss = self.num_layers > 1
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

            x = self.conv_last(h_t[-1])# + x TODO: Removed the residual because process heads mess with input/output shape
            next_frames.append(x)
        if self.use_decouple_loss:
            self.decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        else:
            self.decouple_loss = torch.tensor(0.0).to(self.device)
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


@model_builder.register_emulator('ParflowClmEmulator')
class ParflowClmEmulator(pl.LightningModule):

    def __init__(
        self,
        num_layers,
        num_hidden,
        img_channel,
        act_channel,
        init_cond_channel,
        static_channel,
        out_channel,
        number_depth_layers,
        scale_dict=DEFAULT_SCALERS,
        pressure_first=True,
    ):
        super().__init__()
        # TODO: don't hardcode this
        self.dz = torch.tensor([100.0, 1.0, 0.6, 0.3, 0.1])
        self.fstr_model = ForcedSTRNN(
            num_layers,
            num_hidden,
            img_channel,
            act_channel,
            init_cond_channel,
            static_channel,
            out_channel,
        )
        self.pressure_first = pressure_first
        self.number_depth_layers = number_depth_layers
        self.scale_dict = scale_dict
        self.sat_fun = SaturationHead()
        self.wtd_fun = torch.vmap(WaterTableDepthHead(self.dz))
        self.flow_fun = OverlandFlowHead()


    def forward(
        self, 
        forcings, 
        init_cond, 
        static_inputs, 
        vgn_a,
        vgn_n,
        slope_x,
        slope_y,
        mannings,
        dx=1000.0,
        dy=1000.0,
    ):
        # Predicted dims should be (batch, time, channel, y, x)
        fstr_pred = self.fstr_model(forcings, init_cond, static_inputs)

        if self.pressure_first:
            pressure_pred = fstr_pred[:, :, :self.number_depth_layers, ...].clone()
        else:
            pressure_pred = fstr_pred[:, :, -self.number_depth_layers:, ...].clone()

        # Unscale the predictions
        for i in range(self.number_depth_layers):
            pressure_pred[:, :, i, ...] = self.scale_dict[f'pressure_{i}'].inverse_transform(pressure_pred[:, :, i, ...])

        # Calculate derived quantities
        sat = torch.stack([
            self.sat_fun.forward(pressure_pred[:, t], vgn_a[:, t], vgn_n[:, t])
            for t in range(pressure_pred.shape[1])
        ], dim=1)

        # Don't need to scale saturation since it's already between 0 and 1
        #sat = self.scale_dict['saturation'].transform(sat)
        wtd = torch.stack([
            self.wtd_fun(pressure_pred[:, t], sat[:, t], depth_ax=0)
            for t in range(pressure_pred.shape[1])
        ], dim=1)
        wtd = self.scale_dict['water_table_depth'].transform(wtd)
        flow = torch.stack([self.flow_fun(
            pressure_pred[:, t],
            slope_x[:, t],
            slope_y[:, t],
            mannings[:, t],
            dx, dy, flow_method='OverlandFlow',
        ) for t in range(pressure_pred.shape[1])], dim=1)
        flow = self.scale_dict['streamflow'].transform(flow)

        flow = flow.unsqueeze(2)
        wtd = wtd.unsqueeze(2)

        # Put the full thing back together again
        full_pred = torch.cat([fstr_pred, wtd, flow], dim=2)
        return full_pred

    def log_channel_losses(self, y_hat, y):
        with torch.no_grad():
            for i in range(y.shape[2]):
                loss = self.loss_fun(y_hat[:, :, i, ...], y[:, :, i, ...])
                self.log(f'loss_{i}', loss)

    def training_step(self, train_batch, train_batch_idx):
        forcing, state, params, target, extras = train_batch
        y_hat = self(forcing, state, params, **extras).squeeze()
        loss = self.loss_fun(y_hat, target)
        self.log('train_loss', loss)
        self.log_channel_losses(y_hat, target)
        return loss


    def validation_step(self, val_batch, val_batch_idx):
        forcing, state, params, target, extras = val_batch
        y_hat = self(forcing, state, params, **extras).squeeze()
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
        self.fstr_model.configure_loss(loss_fun)
        self.loss_fun = self.fstr_model.loss_fun



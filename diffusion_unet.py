"""Minimal 1D conditional U-Net from the open-source Diffusion Policy design."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, value):
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        frequency = torch.exp(torch.arange(half, device=value.device) * -scale)
        embedding = value[:, None] * frequency[None]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size=5, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inputs, outputs, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, outputs),
            nn.Mish(),
        )

    def forward(self, value):
        return self.block(value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, inputs, outputs, condition_dim, kernel_size=5):
        super().__init__()
        self.outputs = outputs
        self.blocks = nn.ModuleList(
            (Conv1dBlock(inputs, outputs, kernel_size), Conv1dBlock(outputs, outputs, kernel_size))
        )
        self.condition = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, outputs * 2), nn.Unflatten(-1, (-1, 1))
        )
        self.residual = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, value, condition):
        output = self.blocks[0](value)
        scale, bias = self.condition(condition).reshape(len(value), 2, self.outputs, 1).unbind(1)
        output = self.blocks[1](scale * output + bias)
        return output + self.residual(value)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=(128, 256, 512)):
        super().__init__()
        all_dims = (input_dim, *down_dims)
        pairs = tuple(zip(all_dims[:-1], all_dims[1:]))
        time_dim = 256
        self.time = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        condition_dim = time_dim + global_cond_dim
        self.down = nn.ModuleList()
        for index, (inputs, outputs) in enumerate(pairs):
            self.down.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(inputs, outputs, condition_dim),
                        ConditionalResidualBlock1D(outputs, outputs, condition_dim),
                        nn.Conv1d(outputs, outputs, 3, 2, 1)
                        if index + 1 < len(pairs)
                        else nn.Identity(),
                    )
                )
            )
        middle = all_dims[-1]
        self.middle = nn.ModuleList(
            (
                ConditionalResidualBlock1D(middle, middle, condition_dim),
                ConditionalResidualBlock1D(middle, middle, condition_dim),
            )
        )
        self.up = nn.ModuleList()
        for inputs, outputs in reversed(pairs[1:]):
            self.up.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1D(outputs * 2, inputs, condition_dim),
                        ConditionalResidualBlock1D(inputs, inputs, condition_dim),
                        nn.ConvTranspose1d(inputs, inputs, 4, 2, 1),
                    )
                )
            )
        self.final = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0]), nn.Conv1d(down_dims[0], input_dim, 1)
        )

    def forward(self, sample, timestep, global_cond):
        value = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=value.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        condition = torch.cat((self.time(timestep.expand(len(value))), global_cond), dim=-1)
        skips = []
        for first, second, downsample in self.down:
            value = second(first(value, condition), condition)
            skips.append(value)
            value = downsample(value)
        for block in self.middle:
            value = block(value, condition)
        for first, second, upsample in self.up:
            value = second(first(torch.cat((value, skips.pop()), dim=1), condition), condition)
            value = upsample(value)
        return self.final(value).moveaxis(-1, -2)

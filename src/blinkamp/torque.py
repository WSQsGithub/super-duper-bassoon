"""Supervised auxiliary world-model heads, never an extra policy reward."""

import torch
from torch import nn

from blinkmimic.networks import mlp


class TorqueHeads(nn.Module):
    def __init__(self, input_dim, torque_limits):
        super().__init__()
        self.J = len(torque_limits)
        self.net = mlp(input_dim, [256, 256], 2 * self.J, act=nn.SiLU)
        self.register_buffer('limits', torque_limits.clone())
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normalized_world_model_input):
        y = self.net(normalized_world_model_input)
        actual = self.limits * torch.tanh(y[..., :self.J])
        demand = self.limits * y[..., self.J:]
        return actual, demand

    def loss(self, inputs, actual_target, demand_target):
        actual, demand = self(inputs)
        return (((actual - actual_target) / self.limits).square().mean()
                + ((demand - demand_target) / self.limits).square().mean())

"""Command actor; the critic also sees privileged dynamics and AMP history."""

import torch
from torch import nn

from blinkmimic.mdp import invariant_features
from blinkmimic.networks import RunningNorm, mlp, run_mlp


class CommandActor(nn.Module):
    def __init__(self, obs_dim=93, hist=6, action_dim=29, hidden=(512, 256, 128), action_limit=3., command_encoding='direct'):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        if command_encoding not in ('direct', 'mlp'):
            raise ValueError(command_encoding)
        self.command_encoder = mlp(3, [64, 64], 32) if command_encoding == 'mlp' else None
        self.net = mlp(obs_dim * hist + (32 if self.command_encoder is not None else 3), list(hidden), action_dim)
        self.action_limit = action_limit
        nn.init.normal_(self.net[-1].weight, std=.002)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history, command):
        command = command if self.command_encoder is None else torch.tanh(run_mlp(self.command_encoder, command))
        x = torch.cat((self.norm(history).flatten(1), command), -1)
        return self.action_limit * torch.tanh(run_mlp(self.net, x))


def critic_input(S, command, xi_normalized, amp_history):
    return torch.cat((invariant_features(S), command, xi_normalized, amp_history.flatten(1)), -1)

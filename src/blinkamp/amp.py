"""Least-squares motion prior with an input-differentiable style reward."""

import torch
from torch import nn

from blinkmimic.networks import mlp


class MotionDiscriminator(nn.Module):
    def __init__(self, dim=57, window=10, hidden=(512, 256), reward_mode='quadratic'):
        super().__init__()
        if reward_mode not in ('quadratic', 'softplus'):
            raise ValueError(reward_mode)
        self.reward_mode = reward_mode
        self.net = mlp(dim * window, list(hidden), 1, act=nn.ELU)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        nn.init.uniform_(self.net[-1].weight, -.01, .01)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def initialize_normalizer(self, expert):
        # Fixed expert-derived statistics: never update on normalized or imaginary data.
        self.mean.copy_(expert.mean((0, 1)))
        self.std.copy_(expert.std((0, 1), unbiased=False).clamp(min=.05))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.net(self.normalize(x).flatten(-2)).squeeze(-1)

    def reward(self, x):
        # Deliberately no no_grad: the actor needs d(reward)/d(imagined state).
        if self.reward_mode == 'softplus':
            # Normalize score +1 to reward 1; unlike the quadratic clamp,
            # rejected windows below -1 still provide a recovery gradient.
            return torch.nn.functional.softplus(self(x)) / 1.3132616875182228
        return (1 - .25 * (self(x) - 1).square()).clamp(min=0)

    def loss(self, expert, policy, gradient_penalty=.5):
        real, fake = self(expert.detach()), self(policy.detach())
        classification = (real - 1).square().mean() + (fake + 1).square().mean()
        z = self.normalize(expert).detach().requires_grad_(True)
        score = self.net(z.flatten(1)).sum()
        grad = torch.autograd.grad(score, z, create_graph=True)[0]
        penalty = grad.square().sum((1, 2)).mean()
        return classification + gradient_penalty * penalty, {
            "disc_expert": real.mean().detach(), "disc_policy": fake.mean().detach(),
            "disc_grad_penalty": penalty.detach(), "disc_classification": classification.detach(),
        }


class NegativeReplay:
    """Bounded real-policy history, including the last valid window before a fall."""
    def __init__(self, capacity, window, dim, device):
        self.data = torch.empty(capacity, window, dim, device=device)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    @torch.no_grad()
    def add(self, windows):
        windows = windows.detach()[-self.capacity:]
        n = len(windows)
        if not n:
            return
        first = min(n, self.capacity - self.ptr)
        self.data[self.ptr:self.ptr + first] = windows[:first]
        if first < n:
            self.data[:n - first] = windows[first:]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch):
        if not self.size:
            raise ValueError("No complete real AMP histories collected")
        # Half uniformly from history, half from at most the most recent 4096 windows.
        age = torch.randint(self.size, (batch,), device=self.data.device)
        age[:batch // 2] = torch.randint(min(self.size, 4096), (batch // 2,), device=self.data.device)
        return self.data[(self.ptr - 1 - age) % self.capacity]

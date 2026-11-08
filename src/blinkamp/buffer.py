"""Real transition ring with boundary-aware histories and multi-step dynamics windows."""

import torch


class TransitionReplay:
    keys = ("s", "s1", "a", "a_prev", "xi", "command", "ep_step", "obs", "amp", "done", "term", "timeout",
            "actual_torque", "unclipped_torque")

    def __init__(self, capacity, num_envs):
        self.capacity, self.N = capacity, num_envs
        self.data, self.size, self.ptr = {}, 0, 0

    @torch.no_grad()
    def add(self, transition):
        if not self.data:
            self.data = {k: torch.empty((self.capacity, *transition[k].shape),
                                       dtype=transition[k].dtype, device=transition[k].device) for k in self.keys}
        for key in self.keys:
            self.data[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def rows(self, age):
        return (self.ptr - self.size + age) % self.capacity

    def windows(self, batch, length, holdout=False):
        if self.size < length:
            raise ValueError("Not enough real transitions")
        dev = self.data["s"].device
        t = torch.randint(self.size - length + 1, (batch,), device=dev)
        reserved = max(1, self.N // 16) if self.N > 1 else 0
        low, high = (self.N - reserved, self.N) if holdout else (0, self.N - reserved)
        if low == high:
            raise ValueError("No reserved holdout environments")
        env = torch.randint(low, high, (batch,), device=dev)
        rows = self.rows(t[:, None] + torch.arange(length, device=dev))
        out = {k: v[rows, env[:, None]] for k, v in self.data.items() if k != "obs"}
        done = out["done"].float()
        out["mask"] = torch.cat((torch.ones_like(done[:, :1]), 1 - done[:, :-1]), 1).cumprod(1)
        return out

    def starts(self, batch, history=6, window=10):
        dev = self.data["s"].device
        age = torch.randint(self.size, (batch,), device=dev)
        env = torch.randint(self.N, (batch,), device=dev)
        row = self.rows(age)
        step = self.data["ep_step"][row, env]
        out = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "command", "ep_step")}
        for key, count, name in (("obs", history, "obs_history"), ("amp", window, "amp_history")):
            back = torch.arange(count - 1, -1, -1, device=dev)[None]
            back = torch.minimum(torch.minimum(back, age[:, None]), step[:, None])
            out[name] = self.data[key][self.rows(age[:, None] - back), env[:, None]]
        return out

"""Alternating real-data dynamics/AMP learning and short-horizon differentiable control."""

import copy

import torch
from torch import nn

from blinkmimic.mdp import flatten_state, inv_dim, unflatten_state
from blinkmimic.mirror import Mirror
from blinkmimic.networks import Critic, RunningNorm, WorldModel
from blinkmimic.objective import actor_objective
from blinkmimic.optim import MuonAdam
from .amp import MotionDiscriminator, NegativeReplay
from .buffer import TransitionReplay
from .features import append_history
from .networks import CommandActor, critic_input
from .mdp import style_coefficient
from .torque import TorqueHeads


def optimizer(module, lr):
    return torch.optim.Adam(module.parameters(), lr=lr, fused=next(module.parameters()).is_cuda)


def checked_step(loss, module, opt, clip_value=0.):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(module.parameters(), 1.)
    if clip_value:
        nn.utils.clip_grad_value_(module.parameters(), clip_value)
    finite = bool(torch.isfinite(norm))
    if finite:
        opt.step()
    return norm.detach(), not finite


class Learner:
    def __init__(self, env, cfg):
        self.env, self.cfg, self.features = env, cfg, env.features
        if env.reward_profile != cfg.reward_profile:
            raise ValueError('Environment and learner reward profiles must agree')
        dev, J = env.device, env.spec.num_joints
        RunningNorm.no_sync = True
        env.mdp.cfg.joint_limit = cfg.joint_limit_weight
        env.stand_fraction = cfg.stand_fraction
        self.actor = CommandActor(self.features.obs_dim, cfg.history, command_encoding=cfg.command_encoding,
                                  hidden=tuple(map(int, cfg.actor_hidden.split(',')))).to(dev)
        self.ema = copy.deepcopy(self.actor).requires_grad_(False)
        self.wm = WorldModel(J, env.layout.dim, hidden=tuple(map(int, cfg.wm_hidden.split(','))),
                             structured=True).to(dev)
        self.torque = TorqueHeads(self.wm.in_dim, env.actuation.torque_limits).to(dev)
        critic_dim = inv_dim(J) + 3 + env.layout.dim + cfg.window * env.expert.amp_frames.shape[-1]
        self.critic = Critic(critic_dim).to(dev)
        self.disc = MotionDiscriminator(window=cfg.window, reward_mode=cfg.amp_reward,
                                        normalize=cfg.disc_normalize, activation=cfg.disc_activation).to(dev)
        self.disc.initialize_normalizer(env.expert.sample_windows(4096))
        self.optimizers = {name: optimizer(module, lr) for name, module, lr in (
            ('actor', self.actor, cfg.pi_lr), ('wm', self.wm, cfg.lr),
            ('critic', self.critic, cfg.lr), ('disc', self.disc, cfg.disc_lr), ('torque', self.torque, cfg.lr))}
        if cfg.pi_opt == 'muon':
            mlps = [self.actor.net] + ([self.actor.command_encoder] if self.actor.command_encoder is not None else [])
            self.optimizers['actor'] = MuonAdam(self.actor, mlps, lr=cfg.pi_lr, betas=(.9, .95))
        if cfg.wm_opt == 'muon':
            self.optimizers['wm'] = MuonAdam(self.wm, [self.wm.net], lr=cfg.lr, betas=(.9, .999))
        self.replay = TransitionReplay(cfg.buffer_steps, env.N, 'cpu' if cfg.replay_device == 'cpu' else None)
        self.negatives = NegativeReplay(32768, cfg.window, 57, dev)
        self.mirror = Mirror(env.spec, env.layout.slices, dev)
        self.error_floor = torch.tensor([.003] * 3 + [.005] * 3 + [.02] * 3 + [.05] * 3
                                        + [.003] * J + [.05] * J, device=dev)
        self.nonfinite = 0
        self.sim_steps = 0
        self.imag_step = self._imag_step
        self.wm_loss_fn = self._wm_loss
        self.amp_features = self.features.amp
        self.reward_fn = env.mdp.reward
        if cfg.compile:
            self.imag_step = torch.compile(self._imag_step)
            self.wm_loss_fn = torch.compile(self._wm_loss)
            self.amp_features = torch.compile(self.features.amp)
            self.reward_fn = torch.compile(env.mdp.reward)

    @torch.no_grad()
    def collect(self, steps):
        metrics = {}
        for _ in range(steps):
            self.actor.norm.update(self.env.obs_history[:, -1])
            action = self.actor(self.env.obs_history, self.env.command)
            action = (action + self.cfg.exploration * torch.randn_like(action)).clamp(-3, 3)
            out = self.env.step(action)
            self.replay.add(out)
            self.negatives.add(out['amp_next'][out['amp_valid']])
            # Commands change, but physical states never get force/velocity kicks.
            due = (self.env.ep_step > 0) & (self.env.ep_step % 400 == 0)
            ids = due.nonzero().flatten()
            if len(ids) and not self.env.source_task:
                self.env.command[ids] = self.env.sample_commands(len(ids))
            style = self.disc.reward(out['amp_next'])
            metrics = {**out['metrics'], 'real_task': out['task'], 'real_amp': style,
                       'real_reg': out['reg'], 'fall_fraction': out['term'].float(),
                       'episode_steps': out['ep_step'].float() + 1,
                       'amp_zero_fraction': (style == 0).float()}
            coefficient = (self.cfg.amp_coef if self.cfg.reward_profile == 'physhsi'
                           else style_coefficient(out['command'], self.cfg.amp_coef, self.cfg.amp_when_standing))
            metrics['real_reward'] = self.env.mdp.combined(out['task'], out['reg'], style, coefficient)
            self.sim_steps += self.env.N
        return {k: v.mean().detach() for k, v in metrics.items()}

    def _wm_loss(self, s, s1, a, ap, xi, mask):
        S, truths = unflatten_state(s[:, 0]), unflatten_state(s1)
        losses = []
        std = torch.maximum(self.wm.inc_norm.std(), self.error_floor)
        for t in range(self.cfg.wm_window):
            S = self.wm.step(S, a[:, t], ap[:, t], self.env.layout.normalize(xi[:, t]))
            truth = tuple(x[:, t] for x in truths)
            error = (self.wm.state_error(S, truth) / std).square().mean(-1)
            losses.append(error)
        errors = torch.stack(losses, 1)
        return (errors * mask).sum() / mask.sum().clamp(min=1), errors[:, -1].detach()

    def update_world_model(self):
        w = self.replay.windows(self.cfg.wm_batch, self.cfg.wm_window)
        n = self.cfg.wm_batch // 2
        with torch.no_grad():
            for key in ('s', 's1'):
                w[key][:n] = flatten_state(self.mirror.state(unflatten_state(w[key][:n])))
            for key in ('a', 'a_prev'):
                w[key][:n] = self.mirror.joints(w[key][:n])
            w['xi'][:n] = self.mirror.xi(w['xi'][:n])
            for key in ('actual_torque', 'unclipped_torque'):
                w[key][:n] = self.mirror.joints(w[key][:n])
            S, S1 = unflatten_state(w['s']), unflatten_state(w['s1'])
            self.wm.in_norm.update(self.wm.inputs(S, w['a'], w['a_prev'], self.env.layout.normalize(w['xi'])))
            self.wm.out_norm.update(self.wm.targets(S, S1))
            self.wm.inc_norm.update(self.wm.increments(S, S1))
        loss, last = self.wm_loss_fn(w['s'], w['s1'], w['a'], w['a_prev'], w['xi'], w['mask'])
        _, bad = checked_step(loss, self.wm, self.optimizers['wm'])
        self.nonfinite += bad
        mask = w['mask'][:, -1]
        result = {'wm_loss': loss.detach(), 'wm_last_error': (last * mask).sum() / mask.sum().clamp(min=1)}
        if self.cfg.reward_profile == 'physhsi':
            with torch.no_grad():
                inputs = self.wm.in_norm(self.wm.inputs(S, w['a'], w['a_prev'], self.env.layout.normalize(w['xi'])))
            torque_loss = self.torque.loss(inputs, w['actual_torque'], w['unclipped_torque'])
            _, bad = checked_step(torque_loss, self.torque, self.optimizers['torque'])
            self.nonfinite += bad
            result['torque_loss'] = torque_loss.detach()
        return result

    def update_discriminator(self):
        if not self.negatives.size or self.cfg.amp_coef == 0:
            return {}
        expert = self.env.expert.sample_windows(self.cfg.disc_batch)
        policy = self.negatives.sample(self.cfg.disc_batch)
        loss, metrics = self.disc.loss(expert, policy, self.cfg.grad_penalty)
        _, bad = checked_step(loss, self.disc, self.optimizers['disc'])
        self.nonfinite += bad
        return {'disc_loss': loss.detach(), **metrics}

    def _imag_step(self, S, obs, previous, xi, command):
        action = self.actor(obs, command)
        S1 = self.wm.step(S, action, previous, self.env.layout.normalize(xi))
        o = self.features.noisy(self.features.obs(S1, action, xi))
        return S1, action, append_history(obs, o)

    def _actor_batch(self, batch):
        cfg = self.cfg
        S = S0 = unflatten_state(batch['s'])
        obs, previous = batch['obs_history'], batch['a_prev']
        xi, command, ah0 = batch['xi'], batch['command'], batch['amp_history']
        initial_command = command
        command_steps = []
        # Changing parameters is prohibited here, but input Jacobians stay live.
        frozen = (self.wm, self.disc, self.critic, self.torque)
        for module in frozen:
            module.requires_grad_(False)
        try:
            states, old_states, actions, prevs = [], [], [], []
            for horizon_step in range(cfg.horizon):
                old_states.append(S)
                prevs.append(previous)
                S, previous, obs = self.imag_step(S, obs, previous, xi, command)
                if self.env.source_task:
                    due = ((batch['ep_step'] + horizon_step + 1) % 400 == 0)
                    command = torch.where(due[:, None], self.env.sample_commands(len(command)), command)
                command_steps.append(command)
                states.append(S)
                actions.append(previous)
            H, B = cfg.horizon, len(command)
            S_all = tuple(torch.stack([s[j] for s in states]).flatten(0, 1) for j in range(6))
            features = self.amp_features(S_all).view(H, B, -1).transpose(0, 1)
            seq = torch.cat((ah0[:, 1:], features), 1)
            windows = seq.unfold(1, cfg.window, 1).transpose(-1, -2).permute(1, 0, 2, 3).flatten(0, 1)
            style = self.disc.reward(windows)
            a, ap = torch.stack(actions).flatten(0, 1), torch.stack(prevs).flatten(0, 1)
            commands = torch.stack(command_steps).flatten(0, 1)
            if cfg.reward_profile == 'physhsi':
                old_all = tuple(torch.stack([s[j] for s in old_states]).flatten(0, 1) for j in range(6))
                inputs = self.wm.in_norm(self.wm.inputs(old_all, a, ap, self.env.layout.normalize(xi).repeat(H, 1)))
                actual_torque, unclipped_torque = self.torque(inputs)
                task, reg, _ = self.reward_fn(S_all, commands, a, ap, previous_state=old_all,
                                             actual_torque=actual_torque, unclipped_torque=unclipped_torque)
                coefficient = cfg.amp_coef
            else:
                task, reg, _ = self.reward_fn(S_all, commands, a, ap)
                coefficient = style_coefficient(commands, cfg.amp_coef, cfg.amp_when_standing)
            rewards = self.env.mdp.combined(task, reg, style, coefficient).view(H, B)
            dones = self.env.mdp.soft_terminal(S_all).view(H, B)
            xi_n = self.env.layout.normalize(xi)
            x0 = critic_input(S0, initial_command, xi_n, ah0)
            xs = critic_input(S_all, commands, xi_n.repeat(H, 1), windows)
            if not cfg.critic_norm_after_actor:
                with torch.no_grad():
                    self.critic.norm.update(torch.cat((x0, xs.detach()), 0))
            boot = self.critic(xs[-B:])
            objective, _, _ = actor_objective(rewards, dones, boot, cfg.gamma)
            loss = -objective.mean()
            X = torch.cat((x0, xs), 0).detach().view(H + 1, B, -1)
            return loss, X, rewards.detach(), dones.detach(), {
                'imag_task': task.mean().detach(), 'imag_amp': style.mean().detach(),
                'imag_survival': (1 - dones).mean().detach(), 'imag_reward': rewards.mean().detach()}
        finally:
            for module in frozen:
                module.requires_grad_(True)

    def update_actor_critic(self):
        cfg = self.cfg
        batch = self.replay.starts(cfg.pi_batch, cfg.history, cfg.window)
        chunk = cfg.pi_microbatch or cfg.pi_batch
        opt = self.optimizers['actor']
        opt.zero_grad(set_to_none=True)
        xs, rs, ds, stats = [], [], [], {}
        total_loss = torch.zeros((), device=self.env.device)
        for start in range(0, cfg.pi_batch, chunk):
            part = {k: v[start:start + chunk] for k,v in batch.items()}
            loss, X, rewards, dones, metrics = self._actor_batch(part)
            weight = len(part['s']) / cfg.pi_batch
            (loss * weight).backward()
            total_loss += loss.detach() * weight
            xs.append(X); rs.append(rewards); ds.append(dones)
            for k,v in metrics.items():
                stats[k] = stats.get(k, 0) + weight * v
            del loss
        gn = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        if cfg.pi_clip_value:
            nn.utils.clip_grad_value_(self.actor.parameters(), cfg.pi_clip_value)
        bad = not bool(torch.isfinite(gn))
        if not bad:
            opt.step()
        self.nonfinite += bad
        with torch.no_grad():
            X = torch.cat(xs, 1)
            rewards, dones = torch.cat(rs, 1), torch.cat(ds, 1)
            if cfg.critic_norm_after_actor:
                self.critic.norm.update(X)
            values = self.critic(X)
            ret = values[-1]
            targets = []
            for t in reversed(range(cfg.horizon)):
                ret = rewards[t] + cfg.gamma * (1 - dones[t]) * (
                    (1 - cfg.lam) * values[t + 1] + cfg.lam * ret)
                targets.append(ret)
            target = torch.stack(targets[::-1]).flatten()
            inputs = X[:-1].flatten(0, 1)
        n = len(target)
        mb = n // cfg.critic_minibatches
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=self.env.device) if cfg.critic_minibatches > 1 else torch.arange(n, device=self.env.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                v_loss = (self.critic(inputs[idx]) - target[idx]).square().mean()
                _, bad_v = checked_step(v_loss, self.critic, self.optimizers['critic'])
                self.nonfinite += bad_v
        with torch.no_grad():
            for p, q in zip(self.ema.parameters(), self.actor.parameters()):
                p.lerp_(q, 1 - cfg.ema)
            # Normalization must describe the live policy inputs exactly, not lag behind its mean.
            for p, q in zip(self.ema.buffers(), self.actor.buffers()):
                p.copy_(q)
        return {'actor_loss': total_loss, 'critic_loss': v_loss.detach(), 'actor_gradnorm': gn.detach(), **stats}

    @torch.no_grad()
    def validate_world_model(self, batch=128):
        # Reserved environments are excluded from world-model optimizer batches.
        w = self.replay.windows(batch, self.cfg.wm_window, holdout=True)
        S = unflatten_state(w['s'][:, 0])
        for t in range(self.cfg.wm_window):
            S = self.wm.step(S, w['a'][:, t], w['a_prev'][:, t], self.env.layout.normalize(w['xi'][:, t]))
        truth = unflatten_state(w['s1'][:, -1])
        mask = w['mask'][:, -1]
        avg = lambda x: (x * mask).sum() / mask.sum().clamp(min=1)
        result = {'wm_probe_root_m': avg((S[0] - truth[0]).norm(dim=-1)),
                'wm_probe_joint_rad': avg((S[4] - truth[4]).square().mean(-1).sqrt()),
                'wm_probe_windows': mask.sum()}
        if self.cfg.reward_profile == 'physhsi':
            inputs = self.wm.in_norm(self.wm.inputs(unflatten_state(w['s']), w['a'], w['a_prev'],
                                                   self.env.layout.normalize(w['xi'])))
            actual, demand = self.torque(inputs)
            result['torque_probe_actual_rel_mae'] = ((actual - w['actual_torque']) / self.torque.limits).abs().mean()
            result['torque_probe_demand_rel_mae'] = ((demand - w['unclipped_torque']) / self.torque.limits).abs().mean()
            result['torque_probe_zero_actual_rel_mae'] = (w['actual_torque'] / self.torque.limits).abs().mean()
            result['torque_probe_zero_demand_rel_mae'] = (w['unclipped_torque'] / self.torque.limits).abs().mean()
        return result

"""Exact StyleLoco reward algebra; dynamics/torque acquisition is a separate interface.

Reference: PhysHSI styleloco.py, StyleLoco configs and him_on_policy_runner.py.
No source simulator dependency, torque proxy, extra upright reward or stop-command gating.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysHSIRewardConfig:
    tracking_lin_vel: float = 3.
    tracking_ang_vel: float = 1.5
    tracking_sigma: float = .25
    action_rate: float = -.03
    dof_pos_limits: float = -5.
    dof_vel: float = -1e-4
    dof_acc: float = -1e-7
    torques: float = -1e-4
    dof_vel_limits: float = -1e-3
    torque_limits: float = -.1
    soft_dof_pos_limit: float = .9
    soft_dof_vel_limit: float = .8
    soft_torque_limit: float = .95
    amp_coef: float = .35  # HighKnees recipe; Dinosaur uses .30.
    amp_reward_scale: float = .5


class PhysHSIRewards:
    """Inputs must describe the same real or imagined transition.

    ``actual_torque`` is the applied/clipped actuator torque, while ``unclipped_torque``
    is the controller's pre-limit demand. Neither may be replaced by a pose penalty.
    Previous joint velocity is mandatory for the control-step acceleration cost.
    """

    def __init__(self, q_lo, q_hi, velocity_limits, torque_limits, kp, dt=.02, cfg=None):
        self.cfg = cfg or PhysHSIRewardConfig()
        if dt <= 0 or not 0 <= self.cfg.amp_coef <= 1:
            raise ValueError('Invalid dt or AMP mixture')
        if torch.any(kp <= 0):
            raise ValueError('Positive per-joint PD gains are required for the torque normalization')
        self.dt, self.kp = dt, kp
        mid, half = (q_lo + q_hi) / 2, (q_hi - q_lo) / 2
        self.soft_lo = mid - self.cfg.soft_dof_pos_limit * half
        self.soft_hi = mid + self.cfg.soft_dof_pos_limit * half
        self.velocity_limits, self.torque_limits = velocity_limits, torque_limits

    @staticmethod
    def amp_reward(score):
        return (1 - .25 * (score - 1).square()).clamp(min=0)

    def terms(self, *, command, root_linear_velocity, root_angular_velocity,
              q, qd, previous_qd, action, previous_action, actual_torque, unclipped_torque):
        if actual_torque is None or unclipped_torque is None:
            raise ValueError('PhysHSI alignment requires applied torque and pre-limit torque demand')
        if previous_qd is None:
            raise ValueError('Previous joint velocity is required, including in imagined rollouts')
        c = self.cfg
        ve = (root_linear_velocity[..., :2] - command[..., :2]).square().sum(-1)
        we = (root_angular_velocity[..., 2] - command[..., 2]).square()
        limits = (torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)).sum(-1)
        return {
            'tracking_lin_vel': c.tracking_lin_vel * torch.exp(-ve / c.tracking_sigma),
            'tracking_ang_vel': c.tracking_ang_vel * torch.exp(-we / c.tracking_sigma),
            'action_rate': c.action_rate * (action - previous_action).square().sum(-1),
            'dof_pos_limits': c.dof_pos_limits * limits,
            'dof_vel': c.dof_vel * qd.square().sum(-1),
            'dof_acc': c.dof_acc * ((qd - previous_qd) / self.dt).square().sum(-1),
            'torques': c.torques * (actual_torque / self.kp).square().sum(-1),
            'dof_vel_limits': c.dof_vel_limits * torch.relu(
                qd.abs() - self.velocity_limits * c.soft_dof_vel_limit).sum(-1),
            'torque_limits': c.torque_limits * torch.relu(
                unclipped_torque.abs() - self.torque_limits * c.soft_torque_limit).sum(-1),
        }

    def combine(self, task_reward, discriminator_score):
        """task_reward already includes dt; AMP is added afterward, exactly as in the source runner."""
        c = self.cfg
        return (1 - c.amp_coef) * task_reward + c.amp_coef * c.amp_reward_scale * self.amp_reward(discriminator_score)

    def __call__(self, *, discriminator_score, **transition):
        terms = self.terms(**transition)
        task = self.dt * sum(terms.values())
        return self.combine(task, discriminator_score), task, terms

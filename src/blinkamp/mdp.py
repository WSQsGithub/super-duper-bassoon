"""Reference-free command following; analytic heads shared by reality and imagination."""

from dataclasses import dataclass

import torch

from blinkmimic.motion import CONTROL_DT
from blinkmimic.rotations import mv3
from .features import RootFeatures


def style_coefficient(command, coefficient, amp_when_standing=True):
    if amp_when_standing:
        return coefficient
    # A walking-only expert prior must not require walking under a stop command.
    moving = command.square().sum(-1) > .01
    return coefficient * moving.float()


@dataclass
class RewardConfig:
    velocity_std: float = .5
    yaw_std: float = .5
    action_rate: float = .01
    joint_limit: float = 2.
    vertical_velocity: float = .05
    tilt: float = .1
    min_height: float = .4
    min_up: float = .5


class WalkingMDP:
    def __init__(self, features: RootFeatures, cfg=None):
        self.features, self.cfg = features, cfg or RewardConfig()
        kin = features.kin
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - .9 * half, mid + .9 * half

    def reward(self, S, command, action, previous):
        _, R, v, w, q, _ = S
        c = self.cfg
        vr, wr = mv3(R.transpose(-1, -2), v), mv3(R.transpose(-1, -2), w)
        ve = (vr[..., :2] - command[..., :2]).square().sum(-1)
        we = (wr[..., 2] - command[..., 2]).square()
        task = (2 * torch.exp(-ve / c.velocity_std**2) + torch.exp(-we / c.yaw_std**2)) / 3
        rate = (action - previous).square().sum(-1)
        limits = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        reg = (c.action_rate * rate + c.joint_limit * limits + c.vertical_velocity * v[..., 2].square()
               + c.tilt * (1 - R[..., 2, 2]).square())
        return task, reg, {"velocity_error": ve.sqrt(), "yaw_error": we.sqrt(),
                           "action_rate": rate, "joint_limit": limits}

    def source_margins(self, S):
        R = S[1]
        roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
        pitch = torch.asin((-R[..., 2, 0]).clamp(-1, 1))
        return torch.stack((S[0][..., 2] - .35, .5 - roll.abs(), .85 - (pitch - .25).abs()), -1)

    def terminal(self, S):
        if getattr(self, 'source_task', False):
            return (self.source_margins(S) < 0).any(-1)
        return (S[0][..., 2] < self.cfg.min_height) | (S[1][..., 2, 2] < self.cfg.min_up)

    def soft_terminal(self, S):
        if getattr(self, 'source_task', False):
            return 1 - torch.sigmoid(self.source_margins(S) / .02).prod(-1)
        alive = (torch.sigmoid((S[0][..., 2] - self.cfg.min_height) / .02)
                 * torch.sigmoid((S[1][..., 2, 2] - self.cfg.min_up) / .04))
        return 1 - alive

    def combined(self, task, reg, style_reward, coefficient):
        return CONTROL_DT * ((1 - coefficient) * task + coefficient * style_reward - reg)


class PhysHSIMDP(WalkingMDP):
    """Source reward items and scaling, using the retained controller's real parameters.

    The existing state/observation/termination contract is retained. No legacy
    vertical-velocity or upright reward participates in this profile.
    """
    def __init__(self, features, actuation, coefficient=.35):
        super().__init__(features)
        from .physhsi_rewards import PhysHSIRewardConfig, PhysHSIRewards
        kin = features.kin
        self.rule = PhysHSIRewards(kin.q_lo, kin.q_hi, actuation.velocity_limits,
                                  actuation.torque_limits, actuation.kp,
                                  cfg=PhysHSIRewardConfig(amp_coef=coefficient))

    def reward(self, S, command, action, previous, *, previous_state, actual_torque, unclipped_torque):
        _, R, v, w, q, qd = S
        vr, wr = mv3(R.transpose(-1, -2), v), mv3(R.transpose(-1, -2), w)
        terms = self.rule.terms(command=command, root_linear_velocity=vr, root_angular_velocity=wr,
                                q=q, qd=qd, previous_qd=previous_state[5], action=action, previous_action=previous,
                                actual_torque=actual_torque, unclipped_torque=unclipped_torque)
        task = terms['tracking_lin_vel'] + terms['tracking_ang_vel']
        reg = -sum(value for name, value in terms.items() if not name.startswith('tracking_'))
        metrics = {'velocity_error': (vr[..., :2] - command[..., :2]).norm(dim=-1),
                   'yaw_error': (wr[..., 2] - command[..., 2]).abs(),
                   'action_rate': (action - previous).square().sum(-1),
                   'joint_limit': -terms['dof_pos_limits'] / 5.}
        metrics.update({f'reward_{name}': value * CONTROL_DT for name, value in terms.items()})
        return task, reg, metrics

    def combined(self, task, reg, style_reward, coefficient):
        return (1 - coefficient) * CONTROL_DT * (task - reg) + coefficient * .5 * style_reward

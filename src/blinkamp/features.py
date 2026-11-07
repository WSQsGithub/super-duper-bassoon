"""One pelvis-frame contract for experts, real states, and imagined states."""

import torch

from blinkmimic.rotations import mat_to_6d, mm3, mv3, yaw_mat
from blinkmimic.sensing import obs_noise

KEY_BODIES = (
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "torso_link",
)
AMP_DIM = 57  # 1 + 29 + 5*3 + 3 + 3 + 6


class RootFeatures:
    def __init__(self, kin, robot):
        if robot.sensing.stateful:
            raise ValueError("BlinkAMP currently supports BlinkMimic's stateless G1 sensing only")
        self.kin = kin
        self.default_q = torch.tensor(robot.default_joint_pos_list(), device=kin.device)
        self.key_ids = [robot.tracked_bodies.index(name) for name in KEY_BODIES]
        self.feet = [robot.tracked_bodies.index(name) for name in KEY_BODIES[2:4]]
        self.obs_dim = 3 * robot.num_joints + 6
        # Same ordering and half-widths as BlinkMimic, minus reference orientation.
        self.noise = obs_noise(robot.sensing, robot.num_joints)[:-6].to(kin.device)

    def amp(self, S):
        p, R, v, w, q, qd = S
        bp = self.kin.forward(*S)[0]
        Rt = R.transpose(-1, -2)
        keys = mv3(Rt.unsqueeze(-3), bp[..., self.key_ids, :] - p.unsqueeze(-2)).flatten(-2)
        tilt = mat_to_6d(mm3(yaw_mat(R).transpose(-1, -2), R))
        return torch.cat((p[..., 2:3], q, keys, mv3(Rt, v), mv3(Rt, w), tilt), -1)

    def obs(self, S, a_prev, xi):
        _, R, _, w, q, qd = S
        offset = xi[..., 1:1 + q.shape[-1]]
        return torch.cat((q - self.default_q - offset, qd,
                          mv3(R.transpose(-1, -2), w), -R[..., 2, :], a_prev), -1)

    def noisy(self, obs):
        return obs + (2 * torch.rand_like(obs) - 1) * self.noise


def append_history(history, value):
    return torch.cat((history[:, 1:], value.unsqueeze(1)), 1)

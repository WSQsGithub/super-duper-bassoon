"""Native-60Hz PhysHSI motion preprocessing and fractional-time sampling.

This intentionally preserves source finite differences, provided joint velocities,
10-frame window interpolation and batch-shared 0.9–1.1 time scaling.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from blinkmimic.rotations import quat_to_mat
from .motion import DEFAULT_MANIFEST, REPO
from .physhsi_features import PhysHSIRootFeatures


def source_euler_xyzw(q):
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), -1)


class PhysHSIMotionLibrary:
    def __init__(self, kin, robot, style='dinosaur', manifest=DEFAULT_MANIFEST, window=10):
        if style != 'dinosaur' or window != 10:
            raise ValueError('This source-matched task profile is Dinosaur with the source 10-frame window')
        info = json.loads(Path(manifest).read_text())
        self.entries = [e for e in info['motions'] if e['style'] == style]
        if {e['direction'] for e in self.entries} != {'FW', 'BW', 'SW'}:
            raise ValueError('Dinosaur profile requires all FW/BW/SW clips')
        self.features = PhysHSIRootFeatures(kin, robot)
        self.task_config = json.loads((REPO / 'configs/blinkamp/physhsi_dinosaur.json').read_text())
        self.motion = self
        self.profile, self.style, self.window, self.device = 'physhsi', style, window, kin.device
        self.fps, self.env_fps = 60., 50.
        self.height_shifts, self.joint_projection = [], {}
        order = [info['joint_names'].index(name) for name in robot.joint_names]
        fields = {k: [] for k in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd', 'quat', 'height', 'amp_frames')}
        lengths = []
        for entry in self.entries:
            path = REPO / entry['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            raw = torch.load(path, map_location='cpu', weights_only=True)
            p, quat = raw['base_position'].float(), raw['base_quat'].float()
            q, qd = raw['joint_position'][:, order].float(), raw['joint_velocity'][:, order].float()
            R = quat_to_mat(quat[:, [3, 0, 1, 2]])
            velocity = torch.empty_like(p)
            velocity[:-1] = (p[1:] - p[:-1]) * self.fps
            velocity[-1] = velocity[-2]
            euler = np.unwrap(source_euler_xyzw(quat).numpy(), axis=0)
            angular = torch.empty_like(p)
            angular[:-1] = torch.from_numpy(np.diff(euler, axis=0)) * self.fps
            angular[-1] = angular[-2]
            Rt = R.transpose(-1, -2)
            keys = (Rt[:, None] @ raw['link_position'][:, :5, :, None]).squeeze(-1).flatten(1)
            # Same inverse-heading construction as calc_heading_quat_inv + quat_to_tan_norm.
            from blinkmimic.rotations import yaw_mat
            tilt = yaw_mat(R).transpose(-1, -2) @ R
            rotation = torch.cat((tilt[:, :, 0], tilt[:, :, 2]), -1)
            h = raw['base_height'].float().reshape(-1, 1)
            amp = torch.cat((h, q, keys, (Rt @ velocity[..., None]).squeeze(-1),
                             (Rt @ angular[..., None]).squeeze(-1), rotation), -1)
            values = dict(root_p=p, root_R=R, root_v=velocity, root_w=angular, q=q, qd=qd,
                          quat=quat, height=h, amp_frames=amp)
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f'Invalid {key} in {path}')
                fields[key].append(value)
            lengths.append(len(q))
        # The source unwraps the global concatenated Euler buffer. Preserve the
        # same offsets/float rounding across clip boundaries before differencing.
        unwrapped = np.unwrap(source_euler_xyzw(torch.cat(fields['quat'])).numpy(), axis=0)
        start = 0
        for i, n in enumerate(lengths):
            angular = torch.empty(n, 3)
            angular[:-1] = torch.from_numpy(np.diff(unwrapped[start:start + n], axis=0)) * self.fps
            angular[-1] = angular[-2]
            fields['root_w'][i] = angular
            fields['amp_frames'][i][:, 48:51] = (fields['root_R'][i].transpose(-1, -2) @ angular[..., None]).squeeze(-1)
            start += n
        for key, parts in fields.items():
            setattr(self, key, torch.cat(parts).to(self.device))
        self.names = [Path(e['path']).stem for e in self.entries]
        self.clip_len = torch.tensor(lengths, device=self.device)
        self.clip_start = torch.cat((torch.zeros(1, dtype=torch.long, device=self.device), self.clip_len.cumsum(0)[:-1]))
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(lengths), device=self.device), self.clip_len)
        self.num_frames = sum(lengths)
        self.weights = torch.tensor([e.get('weight', 1.) for e in self.entries], device=self.device)

    def sample_motion_ids(self, batch):
        return torch.multinomial(self.weights, batch, replacement=True)

    def window_indices(self, batch, motion_ids=None, phases=None, rate=None):
        ids = self.sample_motion_ids(batch) if motion_ids is None else motion_ids
        phases = torch.rand(batch, device=self.device) if phases is None else phases
        length = self.clip_len[ids]
        tail = (self.window * self.fps / self.env_fps + 2) / length
        phase = torch.minimum(phases.clamp(min=0), 1 - tail)
        start = self.clip_start[ids] + phase * length
        rate = np.random.uniform(.9, 1.1) if rate is None else rate
        return start[:, None] + torch.arange(self.window, device=self.device) * (self.fps / self.env_fps * rate)

    def sample_windows(self, batch, motion_ids=None, phases=None, rate=None):
        times = self.window_indices(batch, motion_ids, phases, rate)
        lo, hi = times.floor().long(), times.floor().long() + 1
        weight = (times % 1)[..., None]
        return self.amp_frames[lo] * (1 - weight) + self.amp_frames[hi] * weight

    def states_at(self, motion_ids, times):
        floor, ceil = times.floor().long(), times.ceil().long()
        lo, hi = self.clip_start[motion_ids] + floor, self.clip_start[motion_ids] + ceil
        w = (times - floor)[:, None]
        blend = lambda x: x[lo] * (1 - w) + x[hi] * w
        p = blend(self.root_p)
        bias = self.root_p[:, 2:3] - self.height
        p[:, 2:3] = p[:, 2:3] - blend(bias) + .05
        q0, q1 = self.quat[lo], self.quat[hi]
        q0 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q0, q0)
        quat = torch.nn.functional.normalize(q0 * (1 - w) + q1 * w, dim=-1)
        return p, quat_to_mat(quat[:, [3, 0, 1, 2]]), blend(self.root_v), blend(self.root_w), blend(self.q), blend(self.qd)

    def sample_states(self, batch):
        ids = self.sample_motion_ids(batch)
        times = torch.rand(batch, device=self.device) * (self.clip_len[ids] - 1)
        return self.states_at(ids, times)

    def ref_state(self, indices):
        ids = self.clip_id[indices]
        return self.states_at(ids, (indices - self.clip_start[ids]).float())

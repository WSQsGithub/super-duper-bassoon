"""PhysHSI's five points, root-relative height and tangent/normal rotation encoding."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from blinkmimic.rotations import mm3, mv3, yaw_mat
from .features import RootFeatures

SOURCE_KEYPOINTS = ('left_palm_link', 'right_palm_link', 'left_ankle_pitch_link',
                    'right_ankle_pitch_link', 'mid360_link')


def rpy_matrix(rpy):
    x, y, z = rpy
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


class PhysHSIRootFeatures(RootFeatures):
    def __init__(self, kin, robot, reference_urdf=None):
        super().__init__(kin, robot)
        path = reference_urdf or Path(__file__).resolve().parents[2] / 'data/blinkamp/reference/g1_29dof.urdf'
        xml = ET.parse(path).getroot()
        parents = {j.find('child').get('link'): j for j in xml.findall('joint')}
        body_ids, offsets = [], []
        for point in SOURCE_KEYPOINTS:
            link, offset = point, np.zeros(3)
            while link not in kin.body_names:
                joint = parents[link]
                if joint.get('type') != 'fixed':
                    raise ValueError(f'Missing moving ancestor for {point}')
                origin = joint.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                offset = xyz + rpy_matrix(rpy) @ offset
                link = joint.find('parent').get('link')
            body_ids.append(kin.body_names.index(link))
            offsets.append(offset)
        self.point_parents = body_ids
        self.point_offsets = torch.tensor(np.array(offsets), dtype=torch.float32, device=kin.device)
        self.keypoint_names = SOURCE_KEYPOINTS
        self.key_ids = None  # Virtual fixed-link points cannot be represented by tracked-body indices.

    def keypoint_kinematics(self, S):
        P, R, V, W = self.kin.forward(*S, all_bodies=True)
        ids = self.point_parents
        offsets = mv3(R[..., ids, :, :], self.point_offsets)
        positions = P[..., ids, :] + offsets
        velocity = V[..., ids, :] + torch.cross(W[..., ids, :], offsets, dim=-1)
        return positions, velocity

    def amp(self, S):
        p, R, v, w, q, _ = S
        points = self.keypoint_kinematics(S)[0]
        relative = points - p.unsqueeze(-2)
        Rt = R.transpose(-1, -2)
        keys = mv3(Rt.unsqueeze(-3), relative).flatten(-2)
        height = -relative[..., 2:4, 2].amin(-1, keepdim=True)
        tilt = mm3(yaw_mat(R).transpose(-1, -2), R)
        rotation = torch.cat((tilt[..., :, 0], tilt[..., :, 2]), -1)
        return torch.cat((height, q, keys, mv3(Rt, v), mv3(Rt, w), rotation), -1)

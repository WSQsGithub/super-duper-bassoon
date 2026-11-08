import json
from pathlib import Path
import unittest

import numpy as np
import torch

from blinkamp.physhsi_features import SOURCE_KEYPOINTS, PhysHSIRootFeatures
from blinkamp.physhsi_motion import PhysHSIMotionLibrary
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot
from blinkmimic.rotations import so3_exp


class SourceMotionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.robot = get_robot('g1')
        cls.kin = Kinematics(cls.robot, 'cpu')
        cls.lib = PhysHSIMotionLibrary(cls.kin, cls.robot)

    def test_native_frames_and_provided_joint_velocities_are_preserved(self):
        lib = self.lib
        self.assertEqual(len(lib.entries), 3)
        self.assertEqual({e['direction'] for e in lib.entries}, {'FW', 'BW', 'SW'})
        for i, entry in enumerate(lib.entries):
            raw = torch.load(entry['path'], weights_only=True)
            start, end = lib.clip_start[i], lib.clip_end[i] + 1
            self.assertEqual(int(lib.clip_len[i]), len(raw['joint_position']))
            torch.testing.assert_close(lib.q[start:end], raw['joint_position'], rtol=0, atol=0)
            torch.testing.assert_close(lib.qd[start:end], raw['joint_velocity'], rtol=0, atol=0)

    def test_all_five_physical_points_and_feature_conventions_match_experts(self):
        lib = self.lib
        self.assertEqual(lib.features.keypoint_names, SOURCE_KEYPOINTS)
        for i, entry in enumerate(lib.entries):
            raw = torch.load(entry['path'], weights_only=True)
            local = torch.tensor([0, 300, 900, 1496])
            indices = lib.clip_start[i] + local
            S = lib.ref_state(indices)
            points = lib.features.keypoint_kinematics(S)[0]
            expected = raw['link_position'][local, :5] + S[0][:, None]
            torch.testing.assert_close(points, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(lib.features.amp(S), lib.amp_frames[indices], atol=2e-5, rtol=2e-5)
            # Explicitly reject the former torso substitution for head.
            torso = self.kin.forward(*S)[0][:, self.robot.tracked_bodies.index('torso_link')]
            self.assertGreater((points[:, 4] - torso).norm(dim=-1).min().item(), .4)
        S = list(lib.sample_states(2))
        S[1] = torch.eye(3)[None].repeat(2, 1, 1)
        rotation = lib.features.amp(tuple(S))[:, -6:]
        torch.testing.assert_close(rotation, torch.tensor([[1., 0., 0., 0., 0., 1.]]).repeat(2, 1))

    def test_fractional_window_sampling_matches_source_interpolation(self):
        lib = self.lib
        ids = torch.tensor([0, 1, 2])
        phases = torch.tensor([0., .4, 1.])
        rate = 1.1
        actual = lib.sample_windows(3, ids, phases, rate)
        expected = []
        for clip, phase in zip(ids.tolist(), phases.tolist()):
            n, start = int(lib.clip_len[clip]), int(lib.clip_start[clip])
            t0 = start + min(phase, 1 - (10 * 60 / 50 + 2) / n) * n
            rows = []
            for j in range(10):
                t = t0 + j * 60 / 50 * rate
                low = int(np.floor(t)); weight = t % 1
                rows.append((1 - weight) * lib.amp_frames[low] + weight * lib.amp_frames[low + 1])
            expected.append(torch.stack(rows))
        torch.testing.assert_close(actual, torch.stack(expected), atol=3e-4, rtol=3e-4)
        for _ in range(10):
            times = lib.window_indices(1000)
            self.assertTrue((lib.clip_id[times.floor().long()[:, 0]] == lib.clip_id[times.ceil().long()[:, -1]]).all())
            delta = times[:, 1:] - times[:, :-1]
            self.assertTrue((delta >= 1.08 - .001).all() and (delta <= 1.32 + .001).all())

    def test_fractional_reference_state_uses_source_height_nlerp_and_qd(self):
        lib = self.lib
        S = lib.states_at(torch.tensor([0]), torch.tensor([10.25]))
        raw = torch.load(lib.entries[0]['path'], weights_only=True)
        torch.testing.assert_close(S[4][0], .75 * raw['joint_position'][10] + .25 * raw['joint_position'][11])
        torch.testing.assert_close(S[5][0], .75 * raw['joint_velocity'][10] + .25 * raw['joint_velocity'][11])
        self.assertAlmostEqual(S[0][0, 2].item(), (.75 * raw['base_height'][10] + .25 * raw['base_height'][11] + .05).item(), places=6)

    def test_root_invariance_and_extracted_source_preset(self):
        lib = self.lib
        S = lib.sample_states(12)
        p, R, v, w, q, qd = S
        Y = so3_exp(torch.tensor([0., 0., .7]))
        other = (p @ Y.T + torch.tensor([3., -4., .8]), Y @ R, v @ Y.T, w @ Y.T, q, qd)
        torch.testing.assert_close(lib.features.amp(S), lib.features.amp(other), atol=3e-6, rtol=2e-5)
        cfg = json.loads(Path('configs/blinkamp/physhsi_dinosaur.json').read_text())
        self.assertEqual(cfg['initialization']['reference_probability'], .5)
        self.assertEqual(cfg['commands']['ranges']['lin_vel_x'], [-.8, .8])
        self.assertEqual(cfg['keypoints'][-1], 'mid360_link')
        self.assertEqual(cfg['amp_coef'], .3)


if __name__ == '__main__':
    unittest.main()

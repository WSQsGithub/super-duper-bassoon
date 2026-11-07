import unittest

import torch

from blinkamp.features import RootFeatures
from blinkamp.motion import ExpertLibrary
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot
from blinkmimic.rotations import so3_exp


class DataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = get_robot("g1")
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.features = RootFeatures(cls.kin, cls.robot)

    def test_all_local_styles_and_no_cross_clip_windows(self):
        for style in ("walk", "highknees", "dinosaur"):
            lib = ExpertLibrary(self.kin, self.robot, style)
            m = lib.motion
            self.assertTrue((m.q >= self.kin.q_lo - 1e-6).all())
            self.assertTrue((m.q <= self.kin.q_hi + 1e-6).all())
            self.assertEqual(len(lib.joint_projection), len(m.names))
            x = lib.sample_windows(100)
            self.assertEqual(x.shape, (100, 10, 57))
            self.assertTrue(torch.isfinite(x).all())
            starts = lib.sample_indices(1000, complete_window=True)
            self.assertTrue((m.clip_id[starts] == m.clip_id[starts + 9]).all())
            for e, length in zip(lib.entries, m.clip_len.tolist()):
                raw = torch.load(e["path"], weights_only=True)
                self.assertEqual(length, int((len(raw["base_position"]) - 1) / 60 / .02) + 1)

    def test_root_features_ignore_world_xy_and_heading(self):
        S = ExpertLibrary(self.kin, self.robot, "highknees").sample_states(24)
        p, R, v, w, q, qd = S
        Y = so3_exp(torch.tensor([0., 0., 1.2]))
        moved = (p @ Y.T + torch.tensor([4., -2., 0.]), Y @ R, v @ Y.T, w @ Y.T, q, qd)
        torch.testing.assert_close(self.features.amp(S), self.features.amp(moved), atol=2e-6, rtol=2e-5)
        prev = torch.randn_like(q)
        xi = torch.zeros(len(q), 33)
        torch.testing.assert_close(self.features.obs(S, prev, xi), self.features.obs(moved, prev, xi))

    def test_noise_matches_blinkmimic_and_does_not_noise_actions(self):
        from blinkmimic.sensing import obs_noise
        torch.testing.assert_close(self.features.noise, obs_noise(self.robot.sensing, 29)[:-6])
        x = torch.zeros(128, self.features.obs_dim)
        noisy = self.features.noisy(x)
        self.assertTrue((noisy.abs() <= self.features.noise).all())
        self.assertEqual(noisy[:, -29:].abs().max(), 0)


if __name__ == "__main__":
    unittest.main()

"""GPU contract tests, run explicitly during M1b acceptance."""

import unittest

import torch

from blinkamp.env import WalkingEnv
from blinkamp.motion import ExpertLibrary
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import flatten_state, unflatten_state
from blinkmimic.robot import get_robot


@unittest.skipUnless(torch.cuda.is_available(), "GPU simulation required")
class EnvironmentContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.robot = get_robot("g1")
        cls.expert = ExpertLibrary(Kinematics(cls.robot, "cuda:0"), cls.robot, "highknees")
        cls.env = WalkingEnv(cls.robot, cls.expert, 16, randomize=False, episode_steps=2, rsi=0)

    def test_timeout_keeps_terminal_state_and_resets_histories(self):
        env = self.env
        ids = torch.arange(env.N, device=env.device)
        env.reset(ids)
        S = env.state()
        self.assertEqual(S[2].abs().max().item(), 0)
        self.assertEqual(S[3].abs().max().item(), 0)
        a = torch.zeros(env.N, 29, device=env.device)
        first = env.step(a)
        self.assertFalse(first["done"].any())
        second = env.step(a)
        self.assertTrue(second["timeout"].all())
        self.assertTrue((env.ep_step == 0).all())
        self.assertFalse(second["amp_valid"].any())
        self.assertGreater((second["s1"] - second["s"]).abs().max().item(), 0)
        self.assertGreater((second["s1"] - flatten_state(env.state())).abs().max().item(), 0)
        torch.testing.assert_close(env.obs_history[:, 0], env.obs_history[:, -1])
        torch.testing.assert_close(env.amp_history[:, 0], env.amp_history[:, -1])
        # Saved post-step feature matches the saved physical terminal state, not reset.
        torch.testing.assert_close(second["amp_next"][:, -1], env.features.amp(unflatten_state(second["s1"])), atol=1e-5, rtol=1e-4)

    def test_fk_matches_simulator_after_reference_reset(self):
        env = self.env
        env.rsi = 1
        env.reset(torch.arange(env.N, device=env.device))
        p = env.features.kin.forward(*env.state())[0]
        indices = [env.robot.body_names.index(name) for name in self.robot.tracked_bodies]
        torch.testing.assert_close(p, env.robot.data.body_link_pos_w[:, indices], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(env.amp_history[:, -1], env.features.amp(env.state()), atol=1e-5, rtol=1e-4)
        env.rsi = 0


if __name__ == "__main__":
    unittest.main()

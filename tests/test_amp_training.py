"""One real-data alternating update and a checkpoint round-trip on the GPU."""

from pathlib import Path
import tempfile
import unittest

import torch

from blinkamp.buffer import TransitionReplay
from blinkamp.config import TrainConfig
from blinkamp.env import WalkingEnv
from blinkamp.learning import Learner
from blinkamp.motion import ExpertLibrary
from blinkamp.train import checkpoint
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot


class HoldoutContract(unittest.TestCase):
    def test_reserved_environments_are_excluded_from_wm_training(self):
        replay = TransitionReplay(20, 16)
        for t in range(12):
            tr = {k: torch.arange(16)[:, None].float() for k in replay.keys}
            tr.update(done=torch.zeros(16, dtype=torch.bool), term=torch.zeros(16, dtype=torch.bool),
                      timeout=torch.zeros(16, dtype=torch.bool), ep_step=torch.full((16,), t))
            replay.add(tr)
        self.assertTrue((replay.windows(100, 4)['s'] < 15).all())
        self.assertTrue((replay.windows(100, 4, holdout=True)['s'] == 15).all())


@unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
class TrainingContract(unittest.TestCase):
    def test_updates_metrics_and_atomic_checkpoint(self):
        torch.manual_seed(1)
        torch.set_num_threads(4)
        robot = get_robot('g1')
        expert = ExpertLibrary(Kinematics(robot, 'cuda:0'), robot, 'dinosaur', directions=['FW'])
        cfg = TrainConfig(style='dinosaur', num_envs=16, horizon=3, wm_hidden='64,64',
                          wm_batch=16, pi_batch=8, disc_batch=16, warmup_steps=12,
                          buffer_steps=32, compile=False)
        env = WalkingEnv(robot, expert, cfg.num_envs, history=cfg.history)
        learner = Learner(env, cfg)
        metrics = learner.collect(16)
        metrics.update(learner.update_world_model())
        metrics.update(learner.update_discriminator())
        metrics.update(learner.update_actor_critic())
        metrics.update(learner.validate_world_model())
        self.assertTrue(all(torch.isfinite(x).all() for x in metrics.values()))
        self.assertGreater(metrics['actor_gradnorm'].item(), 0)
        self.assertEqual(learner.nonfinite, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            checkpoint(path, learner, cfg, 1, .1)
            self.assertFalse(path.with_suffix('.tmp').exists())
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['iteration'], 1)
            self.assertEqual(set(saved['optimizers']), {'actor', 'wm', 'disc', 'critic', 'torque'})
            for key, value in learner.actor.state_dict().items():
                torch.testing.assert_close(value, saved['models']['actor'][key])

    def test_source_aligned_training_uses_supervised_torque_heads(self):
        torch.manual_seed(9)
        torch.set_num_threads(4)
        robot = get_robot('g1')
        expert = ExpertLibrary(Kinematics(robot, 'cuda:0'), robot, 'walk')
        cfg = TrainConfig(num_envs=16, horizon=3, wm_hidden='64,64', wm_batch=16, pi_batch=8,
                          disc_batch=16, buffer_steps=32, warmup_steps=12, compile=False,
                          reward_profile='physhsi', joint_limit_weight=5., amp_when_standing=True,
                          disc_normalize=False, disc_activation='relu')
        cfg.validate()
        env = WalkingEnv(robot, expert, 16, reward_profile='physhsi')
        learner = Learner(env, cfg)
        metrics = learner.collect(16)
        for _ in range(5):
            metrics.update(learner.update_world_model())
        metrics.update(learner.update_discriminator())
        before = [p.detach().clone() for p in learner.torque.parameters()]
        metrics.update(learner.update_actor_critic())
        for old, new in zip(before, learner.torque.parameters()):
            torch.testing.assert_close(old, new)
        metrics.update(learner.validate_world_model())
        self.assertIn('torque_loss', metrics)
        self.assertIn('reward_torques', metrics)
        self.assertIn('real_reward', metrics)
        self.assertNotIn('reward_tilt', metrics)
        self.assertNotIn('reward_vertical_velocity', metrics)
        self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        self.assertEqual(learner.nonfinite, 0)


if __name__ == '__main__':
    unittest.main()

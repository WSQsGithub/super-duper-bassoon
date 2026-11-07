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
            self.assertEqual(set(saved['optimizers']), {'actor', 'wm', 'disc', 'critic'})
            for key, value in learner.actor.state_dict().items():
                torch.testing.assert_close(value, saved['models']['actor'][key])


if __name__ == '__main__':
    unittest.main()

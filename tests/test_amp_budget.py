import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from blinkamp.config import TrainConfig
from blinkamp.env import WalkingEnv
from blinkamp.motion import ExpertLibrary
from blinkamp.learning import Learner
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot

class BudgetContract(unittest.TestCase):
    def test_preset_matches_released_budget_and_fractional_collection(self):
        config = json.loads(Path('configs/blinkamp/original_budget.json').read_text())
        cfg = TrainConfig(**config)
        cfg.validate()
        self.assertEqual([cfg.vector_steps(i) for i in range(1, 5)], [2,3,2,3])
        self.assertEqual(sum(cfg.vector_steps(i) for i in range(1, 101)), 250)
        self.assertEqual(cfg.command_encoding, 'mlp')
        self.assertEqual(cfg.num_envs, 4096)
        self.assertEqual(cfg.pi_batch * cfg.horizon, 65536)
        self.assertEqual(cfg.wm_batch * cfg.wm_window * cfg.wm_updates, 81920)
        self.assertEqual(cfg.critic_epochs * cfg.critic_minibatches, 20)
        self.assertAlmostEqual(cfg.learning_rates(1)[1], .0016)
        self.assertAlmostEqual(cfg.learning_rates(1701)[0], cfg.learning_rates(1701)[1])
        self.assertEqual(cfg.vector_steps(1) * cfg.num_envs, 8192)
        self.assertEqual(TrainConfig().learning_rates(1), (.0004, .001))

@unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
class UpdateContract(unittest.TestCase):
    def test_cpu_replay_microbatch_actor_equivalence_and_critic_steps(self):
        torch.set_num_threads(4)
        torch.manual_seed(37)
        robot = get_robot('g1')
        expert = ExpertLibrary(Kinematics(robot, 'cuda:0'), robot, 'walk')
        env = WalkingEnv(robot, expert, 16, randomize=False, rsi=0)
        cfg = TrainConfig(num_envs=16, horizon=3, wm_hidden='64,64', wm_batch=16, pi_batch=8,
            disc_batch=16, warmup_steps=12, buffer_steps=32, compile=False,
            critic_epochs=5, critic_minibatches=4, critic_norm_after_actor=True, replay_device='cpu')
        full = Learner(env, cfg)
        full.collect(16)
        self.assertEqual(full.replay.data['s'].device.type, 'cpu')
        self.assertEqual(full.replay.windows(8,4)['s'].device.type, 'cuda')
        batch = full.replay.starts(8, cfg.history, cfg.window)
        batch['ep_step'].fill_(20)
        split_cfg = copy.deepcopy(cfg)
        split_cfg.pi_microbatch = 2
        split = Learner(env, split_cfg)
        for name in ('actor','ema','critic','wm','torque','disc'):
            getattr(split,name).load_state_dict(getattr(full,name).state_dict())
        split.replay = full.replay
        env.features.noisy = lambda x: x
        with patch.object(full.replay,'starts',return_value=batch):
            with patch.object(full.optimizers['actor'],'step',wraps=full.optimizers['actor'].step) as act:
                full.update_actor_critic()
                self.assertEqual(act.call_count,1)
            with patch.object(split.optimizers['actor'],'step',wraps=split.optimizers['actor'].step) as act:
                with patch.object(split.optimizers['critic'],'step',wraps=split.optimizers['critic'].step) as critic:
                    metrics = split.update_actor_critic()
                    self.assertEqual(act.call_count,1)
                    self.assertEqual(critic.call_count,20)
        for x,y in zip(full.actor.parameters(),split.actor.parameters()):
            torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-4)
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        self.assertEqual(split.nonfinite,0)

if __name__ == '__main__':
    unittest.main()

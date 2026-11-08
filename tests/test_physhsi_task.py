import unittest
import torch
from blinkamp.config import TrainConfig
from blinkamp.env import WalkingEnv
from blinkamp.motion import load_expert
from blinkamp.learning import Learner
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot

@unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
class DinosaurTask(unittest.TestCase):
    def test_reset_command_timing_isolation_and_training(self):
        torch.set_num_threads(4)
        torch.manual_seed(13)
        cfg = TrainConfig(task_profile='physhsi_dinosaur', style='dinosaur', direction='all',
            rsi=.5, amp_coef=.3, reward_profile='physhsi', joint_limit_weight=5.,
            disc_normalize=False, disc_activation='relu', num_envs=32, horizon=3,
            wm_hidden='64,64', wm_batch=16, pi_batch=8, disc_batch=16,
            buffer_steps=32, warmup_steps=12, compile=False)
        cfg.validate()
        robot = get_robot('g1')
        expert = load_expert(Kinematics(robot, 'cuda:0'), robot, cfg)
        env = WalkingEnv(robot, expert, cfg.num_envs, randomize=False,
            rsi=cfg.rsi, reward_profile=cfg.reward_profile, amp_coef=cfg.amp_coef,
            task_profile=cfg.task_profile)
        commands = env.sample_commands(10000)
        self.assertTrue((commands[:, :2].abs() <= .8).all())
        self.assertTrue((commands[:, 2].abs() <= .5).all())
        planar = commands[:, :2].norm(dim=-1)
        self.assertTrue(((planar == 0) | (planar > .2)).all())
        self.assertTrue(((commands[:, 2] == 0) | (commands[:, 2].abs() > .2)).all())
        self.assertLess(commands[:, 0].mean().abs().item(), .03)
        env.rsi = 0
        ids = torch.arange(env.N, device=env.device)
        env.reset(ids)
        state = env.state()
        torch.testing.assert_close(state[0][:, 2], torch.full((env.N,), .8, device=env.device))
        self.assertTrue((state[2].abs() <= .5).all())
        self.assertTrue((state[3].abs() <= .5).all())
        self.assertTrue((state[4] >= env.mdp.soft_lo).all())
        self.assertTrue((state[4] <= env.mdp.soft_hi).all())
        torch.testing.assert_close(env.amp_history[:, 0], env.amp_history[:, -1])
        env.command[:] = 9
        env.ep_step[:] = 399
        out = env.step(torch.zeros(env.N, 29, device=env.device), auto_reset=False)
        self.assertTrue((out['command'] == 9).all())
        self.assertTrue((env.command.abs() <= .8).all())
        _, _, metrics = env.mdp.reward(env.state(), env.command, out['a'], out['a_prev'],
            previous_state=state, actual_torque=out['actual_torque'], unclipped_torque=out['unclipped_torque'])
        torch.testing.assert_close(metrics['velocity_error'], out['metrics']['velocity_error'])
        env.reset(ids)
        out = env.step(torch.zeros(env.N, 29, device=env.device))
        self.assertFalse(out['amp_valid'].any())
        learner = Learner(env, cfg)
        learner.collect(16)
        # Exercise a command change inside imagination as well.
        learner.replay.data['ep_step'].fill_(399)
        metrics = learner.update_actor_critic()
        self.assertTrue(all(torch.isfinite(x).all() for x in metrics.values()))
        self.assertEqual(learner.nonfinite, 0)

if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from blinkamp.env import WalkingEnv
from blinkamp.motion import ExpertLibrary
from blinkamp.torque import TorqueHeads
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot


@unittest.skipUnless(torch.cuda.is_available(), 'GPU simulation required')
class TorqueLabelContract(unittest.TestCase):
    def test_real_force_matches_prelimit_kernel_and_reward_has_only_source_terms(self):
        torch.set_num_threads(4)
        r = get_robot('g1')
        expert = ExpertLibrary(Kinematics(r, 'cuda:0'), r, 'walk')
        env = WalkingEnv(r, expert, 16, randomize=False, rsi=0, reward_profile='physhsi')
        for scale in (.1, 3., 15.):
            out = env.step(torch.randn(16, 29, device=env.device) * scale)
            actual, demand = out['actual_torque'], out['unclipped_torque']
            limits = env.actuation.torque_limits
            torch.testing.assert_close(actual, demand.clamp(-limits, limits), atol=2e-4, rtol=1e-5)
            torch.testing.assert_close(actual, env.robot.data.actuator_force, atol=2e-4, rtol=1e-5)
            self.assertTrue(torch.isfinite(out['reg']).all())
            keys = {k.removeprefix('reward_') for k in out['metrics'] if k.startswith('reward_')}
            self.assertEqual(keys, {'tracking_lin_vel', 'tracking_ang_vel', 'action_rate', 'dof_pos_limits',
                                    'dof_vel', 'dof_acc', 'torques', 'dof_vel_limits', 'torque_limits'})


class TorqueHeadContract(unittest.TestCase):
    def test_supervised_heads_are_bounded_and_input_differentiable_when_frozen(self):
        torch.manual_seed(4)
        head = TorqueHeads(12, torch.tensor([10., 20.]))
        inputs = torch.randn(64, 12)
        target = inputs[:, :2].tanh() * head.limits
        demand = inputs[:, :2] * head.limits
        opt = torch.optim.Adam(head.parameters(), lr=.003)
        for _ in range(30):
            loss = head.loss(inputs, target, demand)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), .1)
        head.requires_grad_(False)
        x = inputs.clone().requires_grad_(True)
        actual, demand = head(x)
        self.assertTrue((actual.abs() <= head.limits).all())
        (actual.square().mean() + demand.square().mean()).backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()

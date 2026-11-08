import unittest

import torch

from blinkamp.physhsi_rewards import PhysHSIRewardConfig, PhysHSIRewards


class PhysHSIRewardContract(unittest.TestCase):
    def reward(self, dt=.02):
        return PhysHSIRewards(torch.tensor([-1., -1.]), torch.tensor([1., 1.]),
                              torch.ones(2), torch.full((2,), 10.), torch.full((2,), 2.), dt=dt)

    def transition(self):
        return dict(command=torch.tensor([[.6, 0., 0.]]),
                    root_linear_velocity=torch.tensor([[.6, 0., 0.]]),
                    root_angular_velocity=torch.zeros(1, 3), q=torch.tensor([[.95, 0.]]),
                    qd=torch.tensor([[2., 0.]]), previous_qd=torch.tensor([[1., 0.]]),
                    action=torch.tensor([[1., 0.]]), previous_action=torch.zeros(1, 2),
                    actual_torque=torch.tensor([[4., 0.]]), unclipped_torque=torch.tensor([[12., 0.]]))

    def test_all_enabled_source_terms_and_hand_calculated_total(self):
        r, task, terms = self.reward()(discriminator_score=torch.ones(1), **self.transition())
        expected = {'tracking_lin_vel': 3., 'tracking_ang_vel': 1.5, 'action_rate': -.03,
                    'dof_pos_limits': -.25, 'dof_vel': -.0004, 'dof_acc': -.00025,
                    'torques': -.0004, 'dof_vel_limits': -.0012, 'torque_limits': -.25}
        self.assertEqual(set(terms), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(terms[name].item(), value, places=6)
        self.assertAlmostEqual(task.item(), .02 * sum(expected.values()), places=6)
        self.assertAlmostEqual(r.item(), .65 * .02 * sum(expected.values()) + .175, places=6)

    def test_amp_added_after_dt_and_not_disabled_at_stop(self):
        for dt in (.01, .02, .04):
            rule = self.reward(dt)
            r = rule.combine(torch.zeros(2), torch.tensor([1., -1.]))
            torch.testing.assert_close(r, torch.tensor([.175, 0.]))
        data = self.transition()
        for key in ('command', 'root_linear_velocity', 'qd', 'previous_qd', 'action', 'actual_torque', 'unclipped_torque'):
            data[key].zero_()
        data['q'].zero_()
        r, _, _ = self.reward()(discriminator_score=torch.ones(1), **data)
        self.assertAlmostEqual(r.item(), .65 * .02 * 4.5 + .175, places=6)

    def test_missing_torque_is_rejected_not_approximated(self):
        data = self.transition()
        data['actual_torque'] = None
        with self.assertRaises(ValueError):
            self.reward()(discriminator_score=torch.ones(1), **data)

    def test_negative_task_reward_is_not_clipped(self):
        data = self.transition()
        data['action'].fill_(100)
        r, task, _ = self.reward()(discriminator_score=-torch.ones(1), **data)
        self.assertLess(task.item(), 0)
        self.assertLess(r.item(), 0)

    def test_dinosaur_source_coefficient_and_quadratic_bounds(self):
        c = PhysHSIRewardConfig(amp_coef=.30)
        rule = self.reward()
        rule.cfg = c
        self.assertAlmostEqual(rule.combine(torch.zeros(1), torch.ones(1)).item(), .15, places=6)
        torch.testing.assert_close(rule.amp_reward(torch.tensor([-3., -1., 1., 3.])), torch.tensor([0., 0., 1., 0.]))


if __name__ == '__main__':
    unittest.main()

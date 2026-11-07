import unittest

import torch

from blinkamp.amp import MotionDiscriminator, NegativeReplay
from blinkamp.buffer import TransitionReplay
from blinkamp.features import RootFeatures
from blinkamp.motion import ExpertLibrary
from blinkamp.networks import CommandActor
from blinkmimic.kinematics import Kinematics
from blinkmimic.networks import WorldModel
from blinkmimic.robot import get_robot


class LearningContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_style_gradient_reaches_actor_through_frozen_world_model(self):
        torch.manual_seed(3)
        robot = get_robot('g1')
        kin = Kinematics(robot, 'cpu')
        lib = ExpertLibrary(kin, robot, 'highknees')
        f = RootFeatures(kin, robot)
        S = lib.sample_states(4)
        actor = CommandActor(hidden=(32, 32))
        wm = WorldModel(29, 33, hidden=(32, 32), structured=True)
        disc = MotionDiscriminator(hidden=(32, 32))
        disc.initialize_normalizer(lib.sample_windows(64))
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=.002)  # nonzero learned action Jacobian
        for net in (wm, disc):
            net.requires_grad_(False)
        previous, xi = torch.zeros(4, 29), torch.zeros(4, 33)
        history = f.obs(S, previous, xi)[:, None].expand(-1, 6, -1)
        action = actor(history, torch.tensor([[.6, 0., 0.]]).expand(4, -1))
        predicted = wm.step(S, action, previous, xi)
        window = torch.cat((lib.sample_windows(4)[:, :-1], f.amp(predicted)[:, None]), 1)
        (-disc.reward(window).mean()).backward()
        grad = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 1e-8)
        self.assertTrue(all(p.grad is None for net in (wm, disc) for p in net.parameters()))

    def test_discriminator_learns_without_backpropagating_into_policy_data(self):
        torch.manual_seed(2)
        disc = MotionDiscriminator(dim=3, window=2, hidden=(32, 16))
        expert = torch.randn(64, 2, 3) * .05 + .5
        policy = (torch.randn(64, 2, 3) * .05 - .5).requires_grad_()
        opt = torch.optim.Adam(disc.parameters(), lr=.005)
        for _ in range(35):
            loss, _ = disc.loss(expert, policy)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertIsNone(policy.grad)
        self.assertGreater(disc(expert).mean().item(), .5)
        self.assertLess(disc(policy).mean().item(), -.5)

    def test_ring_histories_do_not_cross_episode_or_ring_boundary(self):
        ring = TransitionReplay(5, 1)
        for t in range(9):
            step = t % 3
            tr = {k: torch.full((1, 1), float(t)) for k in ring.keys}
            tr.update(ep_step=torch.tensor([step]), done=torch.tensor([step == 2]),
                      term=torch.tensor([False]), timeout=torch.tensor([step == 2]))
            ring.add(tr)
        s = ring.starts(1000, history=4, window=4)
        start = s['s'][:, 0]
        earliest = torch.maximum(start - s['ep_step'], torch.tensor(4.))
        self.assertTrue((s['obs_history'][:, :, 0] >= earliest[:, None]).all())
        w = ring.windows(1000, 3)
        for j in range(1, 3):
            self.assertTrue((w['mask'][:, j] == (~w['done'][:, :j].any(1)).float()).all())

    def test_negative_replay_wraps_and_never_keeps_gradient(self):
        ring = NegativeReplay(5, 2, 3, 'cpu')
        ring.add(torch.zeros(4, 2, 3, requires_grad=True))
        ring.add(torch.ones(7, 2, 3, requires_grad=True))
        self.assertEqual(ring.size, 5)
        self.assertFalse(ring.data.requires_grad)
        self.assertTrue((ring.sample(30) == 1).all())

    def test_rejected_windows_can_recover_with_softplus_reward(self):
        disc = MotionDiscriminator(dim=3, window=2, hidden=(16,), reward_mode='softplus')
        with torch.no_grad():
            disc.net[-1].bias.fill_(-3)
        x = torch.randn(8, 2, 3, requires_grad=True)
        self.assertTrue((disc(x) < -1).all())
        reward = disc.reward(x)
        reward.sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue((reward > 0).all())

    def test_stop_commands_do_not_receive_a_walking_only_prior(self):
        from blinkamp.mdp import style_coefficient
        command = torch.tensor([[0., 0., 0.], [.6, 0., 0.]])
        torch.testing.assert_close(style_coefficient(command, .35, False), torch.tensor([0., .35]))

    def test_command_embedding_has_a_live_control_gradient(self):
        actor = CommandActor(hidden=(32,), command_encoding='mlp')
        command = torch.tensor([[0., 0., 0.], [.6, 0., 0.]], requires_grad=True)
        history = torch.zeros(2, 6, 93)
        action = actor(history, command)
        action.sum().backward()
        self.assertGreater(command.grad.abs().sum().item(), 0)
        self.assertGreater((action[0] - action[1]).abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()

import unittest
import torch
from blinkamp.commands import MotionCommands
from blinkamp.physhsi_motion import PhysHSIMotionLibrary
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot

class MotionCommandContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        robot=get_robot('g1');cls.expert=PhysHSIMotionLibrary(Kinematics(robot,'cpu'),robot)
        cls.sampler=MotionCommands(cls.expert)

    def test_one_second_means_preserve_clip_boundaries_and_do_not_clip(self):
        e,s=self.expert,self.sampler
        for i in range(3):
            v=e.amp_frames[e.clip_start[i]:e.clip_end[i]+1][:,[45,46,50]]
            brute=torch.stack([v[t:t+60].mean(0) for t in range(len(v)-59)])
            valid=(brute.abs()<=torch.tensor([.8,.8,.5])).all(-1)
            actual=s.values[s.starts[i]:s.starts[i]+s.lengths[i]]
            torch.testing.assert_close(actual,brute[valid],atol=2e-7,rtol=1e-5)
        self.assertFalse(s.metadata['deadband'])
        self.assertFalse(s.metadata['amp_sampling_changed'])

    def test_equal_clip_probability_and_slow_commands_survive(self):
        torch.manual_seed(731)
        commands,ids=self.sampler.sample(60000,return_motion_ids=True)
        counts=torch.bincount(ids,minlength=3).float()/len(ids)
        self.assertTrue((counts-1/3).abs().max()<.01)
        self.assertTrue((commands.abs()<=torch.tensor([.8,.8,.5])).all())
        slow=commands[:,:2].norm(dim=-1)<.2
        self.assertGreater(slow.float().mean().item(),.05)
        self.assertTrue((commands[slow,:2].abs().sum(-1)>0).all())
        self.assertTrue(torch.isfinite(self.sampler.sample(0)).all())

@unittest.skipUnless(torch.cuda.is_available(),'GPU required')
class MotionCommandEnvironment(unittest.TestCase):
    def test_real_environment_and_imagination_share_sampler(self):
        from blinkamp.config import TrainConfig
        from blinkamp.env import WalkingEnv
        from blinkamp.learning import Learner
        from unittest.mock import patch
        torch.set_num_threads(4);robot=get_robot('g1')
        expert=PhysHSIMotionLibrary(Kinematics(robot,'cuda:0'),robot)
        env=WalkingEnv(robot,expert,16,reward_profile='physhsi',amp_coef=.3,rsi=.5,
            task_profile='physhsi_dinosaur',command_profile='motion_1s')
        env.ep_step[:]=399
        with patch.object(env.command_sampler,'sample',wraps=env.command_sampler.sample) as sample:
            out=env.step(torch.zeros(16,29,device='cuda:0'),auto_reset=False)
            self.assertTrue(sample.called)
        self.assertTrue(torch.isfinite(out['task']).all())
        cfg=TrainConfig(task_profile='physhsi_dinosaur',style='dinosaur',direction='all',rsi=.5,
            amp_coef=.3,command_profile='motion_1s',reward_profile='physhsi',joint_limit_weight=5.,
            disc_normalize=False,disc_activation='relu',num_envs=16,horizon=3,wm_hidden='64,64',
            wm_batch=16,pi_batch=8,disc_batch=16,warmup_steps=12,buffer_steps=32,compile=False)
        learner=Learner(env,cfg);learner.collect(16)
        learner.replay.data['ep_step'].fill_(399)
        with patch.object(env.command_sampler,'sample',wraps=env.command_sampler.sample) as sample:
            metrics=learner.update_actor_critic()
            self.assertTrue(sample.called)
        self.assertTrue(all(torch.isfinite(x).all() for x in metrics.values()))

if __name__=='__main__':unittest.main()

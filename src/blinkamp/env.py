"""G1 command walking on mjlab. No external forces, pushes, or velocity kicks."""

import torch

from mjlab.scene import Scene
from mjlab.sim import Simulation

from blinkmimic.dr import DomainRandomizer, XiLayout
from blinkmimic.env import DECIMATION, PHYSICS_DT, make_scene_cfg, make_sim
from blinkmimic.mdp import flatten_state
from blinkmimic.rotations import mat_to_quat, quat_to_mat
from .features import append_history
from .mdp import PhysHSIMDP, WalkingMDP
from .actuation import ActuationLabels


class WalkingEnv:
    def __init__(self, robot, expert, num_envs=512, device="cuda:0", history=6,
                 randomize=True, episode_steps=750, rsi=.8, command_max=.9,
                 reward_profile='legacy', amp_coef=.35):
        if robot.dr.push_interval_s is not None:
            raise ValueError("External pushes are explicitly unsupported in BlinkAMP")
        self.spec, self.expert, self.features = robot, expert, expert.features
        self.device, self.num_envs, self.N = device, num_envs, num_envs
        self.scene = Scene(make_scene_cfg(robot, num_envs), device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=make_sim(robot), model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        assert tuple(self.robot.joint_names) == robot.joint_names
        self.layout = XiLayout(robot, device)
        self.xi = torch.zeros(num_envs, self.layout.dim, device=device)
        self.randomizer = DomainRandomizer(robot, self.layout, self, eval_mode=not randomize)
        self.randomizer.startup(self.xi)
        self.actuation = ActuationLabels(self)
        self.reward_profile = reward_profile
        if reward_profile not in ('legacy', 'physhsi'):
            raise ValueError(reward_profile)
        self.mdp = (PhysHSIMDP(self.features, self.actuation, amp_coef) if reward_profile == 'physhsi'
                    else WalkingMDP(self.features))
        self.randomize, self.rsi = randomize, rsi
        self.history, self.window = history, expert.window
        self.episode_steps, self.command_max = episode_steps, command_max
        self.stand_fraction = .1
        self.default_q = self.features.default_q
        self.alpha = torch.tensor(robot.action_scale_list(), device=device)
        self.command = torch.zeros(num_envs, 3, device=device)
        self.a_prev = torch.zeros(num_envs, robot.num_joints, device=device)
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.obs_history = torch.zeros(num_envs, history, self.features.obs_dim, device=device)
        self.amp_history = torch.zeros(num_envs, self.window, expert.amp_frames.shape[-1], device=device)
        self.reset(torch.arange(num_envs, device=device))

    def state(self):
        idx = self.robot.indexing
        root = self.sim.data.qpos[:, idx.free_joint_q_adr]
        vel = self.sim.data.qvel[:, idx.free_joint_v_adr]
        R = quat_to_mat(root[:, 3:7])
        return (root[:, :3] - self.scene.env_origins, R, vel[:, :3].clone(),
                (R @ vel[:, 3:6, None]).squeeze(-1),
                self.robot.data.joint_pos.clone(), self.robot.data.joint_vel.clone())

    def sample_commands(self, n):
        command = torch.zeros(n, 3, device=self.device)
        command[:, 0].uniform_(.3, self.command_max)
        command[torch.rand(n, device=self.device) < self.stand_fraction] = 0
        return command

    @torch.no_grad()
    def reset(self, ids, command=None):
        n = len(ids)
        if not n:
            return
        p, R, v, w, q, qd = (x.clone() for x in self.expert.sample_states(n))
        ref = torch.rand(n, device=self.device) < self.rsi
        p[:, :2] = 0
        p[~ref, 2] = .76
        R[~ref] = torch.eye(3, device=self.device)
        v[~ref], w[~ref], qd[~ref] = 0, 0, 0
        q[~ref] = self.default_q
        q.clamp_(self.features.kin.q_lo, self.features.kin.q_hi)
        self.sim.reset(ids)
        self.scene.reset(ids)
        self.robot.write_root_link_pose_to_sim(torch.cat((p + self.scene.env_origins[ids], mat_to_quat(R)), -1), env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=ids)
        self.robot.write_joint_position_to_sim(q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim(qd, env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.randomizer.reset(self.xi, ids)
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0
        self.command[ids] = self.sample_commands(n) if command is None else command
        S = (p, R, v, w, q, qd)
        obs = self.features.obs(S, self.a_prev[ids], self.xi[ids])
        if self.randomize:
            obs = self.features.noisy(obs)
        self.obs_history[ids] = obs[:, None]
        self.amp_history[ids] = self.features.amp(S)[:, None]
        self.sim.forward()

    @torch.no_grad()
    def step(self, action, auto_reset=True):
        action = action.detach()
        S, previous = self.state(), self.a_prev.clone()
        out = {"s": flatten_state(S), "a": action.clone(), "a_prev": previous,
               "xi": self.xi.clone(), "command": self.command.clone(), "ep_step": self.ep_step.clone(),
               "obs": self.obs_history[:, -1].clone(), "amp": self.amp_history[:, -1].clone()}
        target = self.default_q + self.xi[:, 1:1 + action.shape[-1]] + self.alpha * action
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        S1 = self.state()
        actual_torque, unclipped_torque = self.actuation.read()
        self.ep_step += 1
        term = self.mdp.terminal(S1)
        timeout = self.ep_step >= self.episode_steps
        context = (dict(previous_state=S, actual_torque=actual_torque, unclipped_torque=unclipped_torque)
                   if self.reward_profile == 'physhsi' else {})
        task, reg, metrics = self.mdp.reward(S1, self.command, action, previous, **context)
        obs = self.features.obs(S1, action, self.xi)
        if self.randomize:
            obs = self.features.noisy(obs)
        self.obs_history = append_history(self.obs_history, obs)
        self.amp_history = append_history(self.amp_history, self.features.amp(S1))
        out.update(s1=flatten_state(S1), term=term, timeout=timeout, done=term | timeout,
                   task=task, reg=reg, metrics=metrics,
                   amp_next=self.amp_history.clone(), amp_valid=self.ep_step >= self.window - 1,
                   actual_torque=actual_torque, unclipped_torque=unclipped_torque)
        self.a_prev = action.clone()
        # The terminal transition and AMP window have already been copied above.
        if auto_reset:
            self.reset(out["done"].nonzero().flatten())
        return out

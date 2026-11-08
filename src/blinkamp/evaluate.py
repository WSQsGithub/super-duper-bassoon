"""Deterministic, standing-start command evaluation in the real simulator."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import blinkmimic.networks as networks

from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import flatten_state
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mv3
from .amp import MotionDiscriminator
from .config import TrainConfig
from .env import WalkingEnv
from .motion import load_expert
from .networks import CommandActor
from .mdp import style_coefficient


@torch.no_grad()
def evaluate(checkpoint, replicas=32, seconds=20., speeds=None, policy='ema', record=None):
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    ck = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    cfg = TrainConfig(**ck['config'])
    networks.AMP_DTYPE = torch.bfloat16 if cfg.precision == 'bf16' else None
    robot = get_robot('g1')
    expert = load_expert(Kinematics(robot, 'cuda:0'), robot, cfg)
    if speeds is None:
        vectors = ([[0.,0.,0.], [.6,0.,0.], [-.6,0.,0.], [0.,.6,0.], [0.,-.6,0.],
                    [0.,0.,.4], [0.,0.,-.4]] if cfg.task_profile == 'physhsi_dinosaur'
                   else [[v,0.,0.] for v in (0.,.3,.6,.9)])
    else:
        vectors = [[v,0.,0.] for v in speeds]
    count = replicas * len(vectors)
    steps = max(1, round(seconds / .02))
    env = WalkingEnv(robot, expert, count, history=cfg.history, randomize=False, rsi=0, episode_steps=steps + 1,
                     reward_profile=cfg.reward_profile, amp_coef=cfg.amp_coef, task_profile=cfg.task_profile, resample_commands=False)
    env.mdp.cfg.joint_limit = cfg.joint_limit_weight
    actor = CommandActor(env.features.obs_dim, cfg.history, command_encoding=cfg.command_encoding,
                         hidden=tuple(map(int, cfg.actor_hidden.split(',')))).to(env.device).eval()
    actor.load_state_dict(ck['models'][policy])
    disc = MotionDiscriminator(window=cfg.window, reward_mode=cfg.amp_reward,
                               normalize=cfg.disc_normalize, activation=cfg.disc_activation).to(env.device).eval()
    disc.load_state_dict(ck['models']['disc'])
    commands = torch.tensor(vectors, device=env.device).repeat_interleave(replicas, 0)
    env.reset(torch.arange(count, device=env.device), command=commands)
    alive = torch.ones(count, device=env.device, dtype=torch.bool)
    survived_steps = torch.zeros(count, device=env.device)
    sums = {k: torch.zeros(count, device=env.device) for k in
            ('count', 'velocity_sq', 'yaw_sq', 'task', 'amp', 'environment_reward', 'total_reward',
             'slip_sq', 'contacts', 'clearance', 'knee_angle', 'hand_forward')}
    peak_clearance = torch.zeros(count, device=env.device)
    trajectory, actions_record, commands_record = [], [], []
    for t in range(steps):
        action = actor(env.obs_history, env.command)
        action = torch.where(alive[:, None], action, 0.)
        active = alive.clone()
        out = env.step(action, auto_reset=False)
        S = env.state()
        finite = torch.isfinite(flatten_state(S)).all(-1)
        alive &= ~out['term'] & finite
        survived_steps += active
        if t >= min(100, steps // 4):  # report startup separately through survival
            weight = (active & finite).float()
            sums['count'] += weight
            sums['velocity_sq'] += out['metrics']['velocity_error'].square().nan_to_num() * weight
            sums['yaw_sq'] += out['metrics']['yaw_error'].square().nan_to_num() * weight
            sums['task'] += out['task'].nan_to_num() * weight
            style = disc.reward(out['amp_next'])
            sums['amp'] += style.nan_to_num() * weight
            coefficient = (cfg.amp_coef if cfg.reward_profile == 'physhsi'
                           else style_coefficient(commands, cfg.amp_coef, cfg.amp_when_standing))
            sums['environment_reward'] += (.02 * (out['task'] - out['reg'])).nan_to_num() * weight
            sums['total_reward'] += env.mdp.combined(out['task'], out['reg'], style, coefficient).nan_to_num() * weight
            bp, _, bv, _ = env.features.kin.forward(*S)
            feet = env.features.feet
            contact = (bp[:, feet, 2] < .045).float() * weight[:, None]
            sums['slip_sq'] += (bv[:, feet, :2].square().sum(-1).nan_to_num() * contact).sum(-1)
            sums['contacts'] += contact.sum(-1)
            clearance = bp[:, feet, 2].clamp(min=0).mean(-1).nan_to_num()
            sums['clearance'] += clearance * weight
            peak_clearance = torch.maximum(peak_clearance, bp[:, feet, 2].amax(-1).nan_to_num() * weight)
            sums['knee_angle'] += S[4][:, [3, 9]].mean(-1).nan_to_num() * weight
            keys = env.features.keypoint_kinematics(S)[0][:, :2] - S[0][:, None]
            local = mv3(S[1].transpose(-1, -2)[:, None], keys)
            sums['hand_forward'] += local[:, :, 0].mean(-1).nan_to_num() * weight
        if record is not None:
            # Record one 0.6 m/s replica (or the last command), no ghost/phase required.
            index = min(range(len(vectors)), key=lambda j: abs(vectors[j][0] - .6)) * replicas
            trajectory.append(flatten_state(S)[index].cpu().numpy())
            actions_record.append(action[index].cpu().numpy())
            commands_record.append(env.command[index].cpu().numpy())
    results = []
    for j, vector in enumerate(vectors):
        sl = slice(j * replicas, (j + 1) * replicas)
        n = sums['count'][sl].sum().item()
        mean = lambda name: sums[name][sl].sum().item() / n if n else None
        contacts = sums['contacts'][sl].sum().item()
        results.append({'command': vector, 'survival_rate': alive[sl].float().mean().item(),
                        'mean_survival_seconds': survived_steps[sl].mean().item() * .02,
                        'velocity_rmse_m_s': mean('velocity_sq') ** .5 if n else None,
                        'yaw_rmse_rad_s': mean('yaw_sq') ** .5 if n else None,
                        'task_reward': mean('task'), 'amp_reward_diagnostic': mean('amp'),
                        'environment_reward_mean': mean('environment_reward'), 'total_reward_mean': mean('total_reward'),
                        'foot_slip_rms_m_s': (sums['slip_sq'][sl].sum().item() / contacts) ** .5 if contacts else None,
                        'mean_ankle_clearance_m': mean('clearance'),
                        'peak_ankle_clearance_m': peak_clearance[sl].max().item(),
                        'mean_knee_angle_rad': mean('knee_angle'), 'mean_hand_forward_m': mean('hand_forward')})
    if record is not None:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(record, states=np.stack(trajectory), dt=.02, style=cfg.style,
                            checkpoint=str(checkpoint), policy=policy, actions=np.stack(actions_record), commands=np.stack(commands_record))
    return {'checkpoint': str(checkpoint), 'style': cfg.style, 'iteration': ck['iteration'], 'reward_profile': cfg.reward_profile,
            'policy': policy, 'replicas_per_command': replicas, 'seconds': seconds,
            'reset': ('source default pose, joint offsets ±.1, root velocities ±.5'
                      if cfg.task_profile == 'physhsi_dinosaur' else 'standing, zero velocity'), 'noise': False, 'external_disturbances': False,
            'coordinate_frame': 'pelvis/root', 'results': results,
            'task_profile': cfg.task_profile,
            'note': 'Fixed seed and fixed commands; source task replicas have independently sampled initialization. '
                    'AMP scores from different discriminators are not comparable style-quality metrics.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--replicas', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--speeds', type=float, nargs='+', default=None)
    parser.add_argument('--policy', choices=['actor', 'ema'], default='ema')
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    if args.replicas < 1 or args.seconds <= 0:
        parser.error('replicas and seconds must be positive')
    report = evaluate(args.checkpoint, args.replicas, args.seconds, args.speeds, args.policy, args.record)
    out = args.out or args.checkpoint.parent / f'eval_{args.checkpoint.stem}_{args.policy}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

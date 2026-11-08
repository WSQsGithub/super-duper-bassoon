"""Train root-frame command walking using real-data dynamics and a differentiable AMP prior."""

import argparse
from dataclasses import fields
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot
import blinkmimic.rotations as rotations
import blinkmimic.networks as networks
from .config import TrainConfig
from .env import WalkingEnv
from .learning import Learner
from .motion import DEFAULT_MANIFEST, load_expert


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON defaults; explicit CLI flags override')
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Warm restart: replay and physics solver state are refilled')
    parser.add_argument('--wm-init', type=Path, help='Initialize compatible structured WM only, never the tracking actor')
    for field in fields(TrainConfig):
        kw = {'default': field.default}
        if field.type is bool:
            kw['action'] = argparse.BooleanOptionalAction
        else:
            kw['type'] = field.type
        parser.add_argument('--' + field.name.replace('_', '-'), **kw)
    # Historical checkpoint loading still defaults to legacy; all new CLI runs use source-aligned rewards.
    parser.set_defaults(task_profile='physhsi_dinosaur', style='dinosaur', direction='all', rsi=.5, amp_coef=.3, reward_profile='physhsi', joint_limit_weight=5., amp_reward='quadratic',
                        amp_when_standing=True, disc_normalize=False, disc_activation='relu')
    known, _ = parser.parse_known_args()
    if known.config:
        defaults = json.loads(known.config.read_text())
        unknown = set(defaults) - {f.name for f in fields(TrainConfig)}
        if unknown:
            parser.error(f'Unknown configuration keys: {unknown}')
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    cfg.validate()
    if args.resume and args.wm_init:
        parser.error('--resume and --wm-init are mutually exclusive')
    return args, cfg


def checkpoint(path, learner, cfg, iteration, wall_time):
    path = Path(path)
    data = {'schema': 2, 'config': cfg.asdict(), 'iteration': iteration, 'wall_time': wall_time,
            'sim_steps': learner.sim_steps, 'nonfinite': learner.nonfinite,
            'models': {k: getattr(learner, k).state_dict() for k in ('actor', 'ema', 'wm', 'disc', 'critic', 'torque')},
            'optimizers': {k: v.state_dict() for k, v in learner.optimizers.items()},
            'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                    'numpy': np.random.get_state(), 'python': random.getstate()},
            'resume_semantics': 'Warm restart: replay refilled and simulator reinitialized; not bitwise continuation'}
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    os.replace(temp, path)


def main():
    args, cfg = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA GPU is required')
    if (args.log_dir / 'args.json').exists() and not args.resume:
        raise FileExistsError('Run already exists; use a fresh log directory or --resume')
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rotations.ELEMENTWISE = True
    networks.AMP_DTYPE = torch.bfloat16 if cfg.precision == 'bf16' else None
    args.log_dir.mkdir(parents=True, exist_ok=True)
    robot = get_robot('g1')
    kin = Kinematics(robot, 'cuda:0')
    expert = load_expert(kin, robot, cfg)
    env = WalkingEnv(robot, expert, cfg.num_envs, history=cfg.history, rsi=cfg.rsi, command_max=cfg.command_max,
                     reward_profile=cfg.reward_profile, amp_coef=cfg.amp_coef, task_profile=cfg.task_profile)
    learner = Learner(env, cfg)
    start_iteration, wall_offset = 0, 0.
    if args.wm_init:
        ck = torch.load(args.wm_init, map_location=env.device, weights_only=False)
        if 'models' in ck:
            source_cfg, weights = ck['config'], ck['models']['wm']
            if source_cfg['wm_hidden'] != cfg.wm_hidden:
                raise ValueError('WM hidden dimensions do not match --wm-init')
        else:
            if not ck['args'].get('wm_structured') or ck['args'].get('robot', 'g1') != 'g1':
                raise ValueError('Only a structured G1 world model can initialize this run')
            weights = ck['wm']
        learner.wm.load_state_dict(weights, strict=True)
        print(f'Initialized world model only from {args.wm_init}', flush=True)
    if args.resume:
        ck = torch.load(args.resume, map_location=env.device, weights_only=False)
        old = TrainConfig(**ck['config']).asdict()
        if cfg.reward_profile == 'physhsi' and ('torque' not in ck['models'] or 'torque' not in ck['optimizers']):
            raise ValueError('Aligned resume requires trained torque heads and their optimizer; use --wm-init for old models')
        allowed = {'iterations', 'ckpt_every', 'log_every', 'compile', 'pi_microbatch', 'replay_device'}
        changed = {k for k, v in cfg.asdict().items() if k not in allowed and old[k] != v}
        if changed:
            raise ValueError(f'Resume config differs: {sorted(changed)}')
        for key, value in ck['models'].items():
            getattr(learner, key).load_state_dict(value)
        for key, value in ck['optimizers'].items():
            learner.optimizers[key].load_state_dict(value)
        start_iteration, wall_offset = ck['iteration'], ck['wall_time']
        learner.sim_steps, learner.nonfinite = ck['sim_steps'], ck['nonfinite']
        torch.set_rng_state(ck['rng']['torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ck['rng']['cuda']])
        np.random.set_state(ck['rng']['numpy'])
        random.setstate(ck['rng']['python'])
        print('Warm restart: restoring networks, optimizers and RNG; refilling replay, reinitializing physics.', flush=True)
    (args.log_dir / ('args_resume.json' if args.resume else 'args.json')).write_text(json.dumps(cfg.asdict(), indent=2) + '\n')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {'git_revision': revision, 'gpu': torch.cuda.get_device_name(0),
                'packages': {x: importlib.metadata.version(x) for x in ('torch', 'mjlab', 'mujoco', 'warp-lang')},
                'manifest_sha256': hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
                'expert_files': [x['path'] for x in expert.entries], 'height_shifts': expert.height_shifts,
                'expert_joint_projection': expert.joint_projection,
                'coordinate_frame': 'pelvis/root', 'external_disturbances': False,
                'wm_init': str(args.wm_init) if args.wm_init else None}
    metadata.update(task_profile=cfg.task_profile, amp_keypoints=list(expert.features.keypoint_names),
                    amp_reset_isolation=True, task_configuration=getattr(expert, 'task_config', None))
    metadata['reward_profile'] = cfg.reward_profile
    metadata['torque_labels'] = 'measured applied actuator force + exact MuJoCo pre-force-limit signal'
    if cfg.reward_profile == 'physhsi':
        from dataclasses import asdict
        metadata['reward_configuration'] = asdict(env.mdp.rule.cfg)
    (args.log_dir / ('environment_resume.json' if args.resume else 'environment.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    writer = SummaryWriter(str(args.log_dir / 'tensorboard'), purge_step=start_iteration + 1 if args.resume else None)
    t0 = time.perf_counter()
    learner.collect(cfg.warmup_steps)
    fields_log = ['iteration', 'wall_seconds', 'sim_steps', 'iterations_per_second',
                  'real_task', 'real_amp', 'real_reg', 'velocity_error', 'yaw_error', 'action_rate', 'joint_limit',
                  'fall_fraction', 'episode_steps', 'amp_zero_fraction', 'wm_loss', 'wm_last_error',
                  'disc_loss', 'disc_expert', 'disc_policy', 'disc_grad_penalty', 'disc_classification',
                  'actor_loss', 'critic_loss', 'actor_gradnorm', 'imag_task', 'imag_amp', 'imag_survival',
                  'wm_probe_root_m', 'wm_probe_joint_rad', 'wm_probe_windows', 'nonfinite', 'gpu_peak_gb',
                  'lr', 'pi_lr', 'collect_vector_steps', 'real_reward', 'imag_reward', 'torque_loss', 'torque_probe_actual_rel_mae',
                  'torque_probe_demand_rel_mae', 'torque_probe_zero_actual_rel_mae', 'torque_probe_zero_demand_rel_mae']
    fields_log += ['reward_' + name for name in ('tracking_lin_vel', 'tracking_ang_vel', 'action_rate',
                  'dof_pos_limits', 'dof_vel', 'dof_acc', 'torques', 'dof_vel_limits', 'torque_limits')]
    csv_path = args.log_dir / 'train_log.csv'
    exists = csv_path.exists()
    with csv_path.open('a', newline='') as stream:
        csv_writer = csv.DictWriter(stream, fields_log)
        if not exists:
            csv_writer.writeheader()
        last_t, last_it = time.perf_counter(), start_iteration
        for it in range(start_iteration + 1, cfg.iterations + 1):
            base_lr, actor_lr = cfg.learning_rates(it)
            for name in ('wm', 'critic', 'torque', 'actor'):
                for group in learner.optimizers[name].param_groups:
                    group['lr'] = actor_lr if name == 'actor' else base_lr
            metrics = learner.collect(cfg.vector_steps(it))
            for _ in range(cfg.wm_updates):
                metrics.update(learner.update_world_model())
            for _ in range(cfg.disc_updates):
                metrics.update(learner.update_discriminator())
            warmup = max(cfg.wm_warmup_iters, cfg.torque_warmup_iters) if cfg.reward_profile == 'physhsi' else cfg.wm_warmup_iters
            if it > warmup:
                metrics.update(learner.update_actor_critic())
            if it % cfg.log_every == 0 or it == cfg.iterations:
                metrics.update(learner.validate_world_model())
                now = time.perf_counter()
                row = {k: float(v) for k, v in metrics.items()}
                row.update(lr=base_lr, pi_lr=actor_lr, collect_vector_steps=cfg.vector_steps(it), iteration=it, wall_seconds=now - t0 + wall_offset, sim_steps=learner.sim_steps,
                           iterations_per_second=(it - last_it) / (now - last_t), nonfinite=learner.nonfinite,
                           gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30)
                csv_writer.writerow(row)
                stream.flush()
                for key, value in row.items():
                    if key not in ('iteration',):
                        writer.add_scalar(key, value, it)
                writer.flush()
                print(f"it {it} | {row['wall_seconds']/60:.2f} min | real task {row['real_task']:.3f} "
                      f"amp {row['real_amp']:.3f} | vel err {row['velocity_error']:.3f} | "
                      f"WM {row['wm_loss']:.3f} | {row['iterations_per_second']:.2f} it/s | nonfinite {learner.nonfinite}", flush=True)
                last_t, last_it = now, it
            if it % cfg.ckpt_every == 0 or it == cfg.iterations or it == 3500:
                elapsed = time.perf_counter() - t0 + wall_offset
                checkpoint(args.log_dir / f'ckpt_{it:06d}.pt', learner, cfg, it, elapsed)
                checkpoint(args.log_dir / 'last.pt', learner, cfg, it, elapsed)
    writer.close()


if __name__ == '__main__':
    main()

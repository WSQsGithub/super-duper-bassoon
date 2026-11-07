"""Discriminator-independent pose-window diagnostics using a shared expert normalization.

This is a descriptive distribution comparison, not a perceptual quality or robustness certificate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blinkamp.motion import ExpertLibrary
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import unflatten_state
from blinkmimic.robot import get_robot


def gaussian_mmd(x, y, bandwidth_sq):
    k = lambda a, b: torch.exp(-torch.cdist(a, b).square() / (2 * bandwidth_sq))
    return (k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()).clamp(min=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Pilot root containing taskonly/walk/highknees/dinosaur rollouts')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    robot = get_robot('g1')
    kin = Kinematics(robot, 'cpu')
    libs = {style: ExpertLibrary(kin, robot, style, directions=None if style == 'walk' else ['FW'])
            for style in ('walk', 'highknees', 'dinosaur')}
    # Drop root linear/angular velocities so commanded speed does not dominate pose style.
    columns = torch.tensor(list(range(45)) + list(range(51, 57)))
    frames = torch.cat([lib.amp_frames[:, columns] for lib in libs.values()])
    mean, std = frames.mean(0), frames.std(0, unbiased=False).clamp(min=.05)
    expert = {name: ((lib.sample_windows(256)[:, :, columns] - mean) / std).flatten(1)
              for name, lib in libs.items()}
    pool = torch.cat(list(expert.values()))
    distances = torch.pdist(pool).square()
    bandwidth_sq = distances.median().clamp(min=1e-6)
    expert_descriptors = {}
    for style, lib in libs.items():
        m, feet = lib.motion, lib.features.feet
        expert_descriptors[style] = {
            'knee_angle_mean_rad': float(m.q[:, [3, 9]].mean()),
            'knee_angle_95pct_rad': float(torch.quantile(m.q[:, [3, 9]], .95)),
            'ankle_clearance_95pct_m': float(torch.quantile(m.body_p[:, feet, 2], .95)),
            'hand_forward_mean_m': float(lib.amp_frames[:, [30, 33]].mean()),
        }
    results = {}
    for name in ('taskonly', 'walk', 'highknees', 'dinosaur'):
        path = args.root / name / 'rollout.npz'
        if not path.exists():
            results[name] = {'available': False}
            continue
        with np.load(path) as raw:
            states = torch.tensor(raw['states'])
            dt = float(raw['dt'])
        S = unflatten_state(states)
        good = torch.isfinite(states).all(-1) & (S[0][:, 2] >= .4) & (S[1][:, 2, 2] >= .5)
        bad = (~good).nonzero().flatten()
        end = int(bad[0]) if len(bad) else len(states)
        start = min(100, len(states) // 4)
        if end - start < 10:
            results[name] = {'available': False, 'reason': 'falls before enough steady-state motion', 'valid_seconds': end * dt}
            continue
        S = tuple(x[start:end] for x in S)
        features = libs['walk'].features.amp(S)
        sequence = features[:, columns].unfold(0, 10, 1).transpose(-1, -2)
        ids = torch.randint(len(sequence), (256,))
        sampled = ((sequence[ids] - mean) / std).flatten(1)
        bp = kin.forward(*S)[0]
        feet = libs['walk'].features.feet
        results[name] = {'available': True, 'valid_seconds': end * dt,
                         'pose_window_mmd': {style: float(gaussian_mmd(sampled, examples, bandwidth_sq))
                                             for style, examples in expert.items()},
                         'knee_angle_mean_rad': float(S[4][:, [3, 9]].mean()),
                         'knee_angle_95pct_rad': float(torch.quantile(S[4][:, [3, 9]], .95)),
                         'ankle_clearance_95pct_m': float(torch.quantile(bp[:, feet, 2], .95)),
                         'hand_forward_mean_m': float(features[:, [30, 33]].mean())}
    report = {'normalization': 'Shared all-style expert pose mean/std, no root velocities',
              'features': '10 pelvis-frame pose frames; same FK, indices, normalization and RBF bandwidth for every style',
              'expert_window_samples': 256, 'rbf_bandwidth_squared': float(bandwidth_sq), 'runs': results,
              'expert_descriptors': expert_descriptors,
              'limitations': 'One recorded 0.6 m/s rollout per policy; training experts reused as references. '
                             'MMD is descriptive, not a held-out style-generalization or perceptual quality result.'}
    out = args.root / 'style_diagnostics.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

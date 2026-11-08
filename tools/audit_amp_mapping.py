"""Independent URDF-FK, motion-column and policy-posture audit (CPU only)."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from blinkamp.config import TrainConfig
from blinkamp.features import append_history
from blinkamp.motion import ExpertLibrary
from blinkamp.networks import CommandActor
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import unflatten_state
from blinkmimic.robot import get_robot
from blinkmimic.rotations import quat_to_mat


def read_mapping(path):
    pairs = [(int(line.split()[0]), line.split()[1]) for line in Path(path).read_text().splitlines() if line.strip()]
    if sorted(i for i, _ in pairs) != list(range(len(pairs))) or len(set(n for _, n in pairs)) != len(pairs):
        raise ValueError('Mapping needs unique names and contiguous explicit column IDs')
    return [name for _, name in sorted(pairs)]


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def urdf_fk(root, angles):
    """NumPy URDF tree FK, independent of BlinkMimic's FK and MuJoCo importer."""
    joints = list(root.findall('joint'))
    children = {j.find('child').get('link') for j in joints}
    roots = {x.get('name') for x in root.findall('link')} - children
    poses = {name: (np.zeros(3), np.eye(3)) for name in roots}
    while joints:
        progress = False
        for joint in joints[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in poses:
                continue
            origin = joint.find('origin')
            attrs = origin.attrib if origin is not None else {}
            xyz = np.fromstring(attrs.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(attrs.get('rpy', '0 0 0'), sep=' ')
            rotation = axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1]) @ axis_rotation([1, 0, 0], rpy[0])
            if joint.get('type') in ('revolute', 'continuous'):
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                rotation = rotation @ axis_rotation(axis, angles[joint.get('name')])
            elif joint.get('type') != 'fixed':
                raise ValueError(f'Unsupported joint type: {joint.get("type")}')
            p, R = poses[parent]
            poses[child] = (p + R @ xyz, R @ rotation)
            joints.remove(joint)
            progress = True
        if not progress:
            raise ValueError('Disconnected or cyclic URDF')
    return poses


def compare_fk(source_root, model, source_names, imported_names, poses):
    joints = {j.get('name'): j for j in source_root.findall('joint')}
    data = mujoco.MjData(model)
    pos_error, rotation_error = 0., 0.
    for raw in poses:
        reference = urdf_fk(source_root, dict(zip(source_names, raw)))
        data.qpos[:] = model.qpos0
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        for column, name in enumerate(imported_names):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = raw[column]
        mujoco.mj_forward(model, data)
        for name in source_names:
            body = joints[name].find('child').get('link')
            idx = model.body(body).id
            p, R = reference[body]
            pos_error = max(pos_error, float(np.linalg.norm(data.xpos[idx] - p)))
            rotation_error = max(rotation_error, float(np.abs(data.xmat[idx].reshape(3, 3) - R).max()))
    return {'max_body_position_error_m': pos_error, 'max_rotation_matrix_error': rotation_error}


def raw_keypoint_audit(source, mapping, entries):
    """Check raw column semantics against the separately supplied hand/foot positions."""
    bodies = ('left_palm_link', 'right_palm_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link')
    worst, feet_worst, style_worst, hand_delta_worst = 0., 0., 0., 0.
    per_clip = {}
    for entry in entries:
        raw = torch.load(entry['path'], map_location='cpu', weights_only=True)
        clip_error = 0.
        for t in np.linspace(0, len(raw['joint_position']) - 1, 3).astype(int):
            fk = urdf_fk(source, dict(zip(mapping, raw['joint_position'][t].numpy())))
            local = np.stack([fk[body][0] for body in bodies])
            R = quat_to_mat(raw['base_quat'][t, [3, 0, 1, 2]]).numpy()
            expected = local @ R.T
            delta = expected - raw['link_position'][t, :4].numpy()
            errors = np.linalg.norm(delta, axis=-1)
            clip_error = max(clip_error, float(errors.max()))
            feet_worst = max(feet_worst, float(errors[2:].max()))
            hand_delta_worst = max(hand_delta_worst, float(np.linalg.norm(delta[0] - delta[1])))
        per_clip[Path(entry['path']).name] = clip_error
        worst = max(worst, clip_error)
        if entry['style'] != 'walk':
            style_worst = max(style_worst, clip_error)
    return {'clips_checked': len(entries), 'frames_per_clip': 3, 'points_per_frame': 4,
            'max_raw_hand_foot_position_error_m': worst,
            'max_style_clip_point_error_m': style_worst, 'max_foot_error_m': feet_worst,
            'max_left_right_hand_residual_difference_m': hand_delta_worst,
            'per_clip_errors_m': per_clip,
            'note': 'Ordinary clips have a shared 2-4mm hand translation residual. Foot points and style clips match '
                    'at numerical precision. BlinkAMP ignores these stored link_position values and rebuilds FK.'}


@torch.no_grad()
def policy_audit(run, robot, kin):
    ck = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    cfg = TrainConfig(**ck['config'])
    with np.load(run / 'rollout.npz') as raw:
        states = torch.tensor(raw['states'])
        dt = float(raw['dt'])
        policy = str(raw['policy'])
    expert = ExpertLibrary(kin, robot, cfg.style, directions=None if cfg.style == 'walk' or cfg.direction == 'all' else [cfg.direction])
    features = expert.features
    actor = CommandActor(features.obs_dim, cfg.history, command_encoding=cfg.command_encoding)
    actor.load_state_dict(ck['models'][policy])
    # Reconstruct exactly the deterministic actor history of the recorded 0.6m/s evaluation.
    q0 = features.default_q[None]
    initial = (torch.tensor([[0., 0., .76]]), torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3), q0, torch.zeros_like(q0))
    xi, previous = torch.zeros(1, 33), torch.zeros(1, 29)
    xi[:, 0] = 1
    history = features.obs(initial, previous, xi)[:, None].repeat(1, cfg.history, 1)
    command = torch.tensor([[.6, 0., 0.]])
    actions = []
    for state in states:
        action = actor(history, command)
        actions.append(action[0])
        observation = features.obs(unflatten_state(state[None]), action, xi)
        history = append_history(history, observation)
    actions = torch.stack(actions)
    start = min(100, len(states) // 4)
    q, a = states[start:, 13:42], actions[start:]
    low, high = torch.quantile(expert.motion.q, .01, dim=0), torch.quantile(expert.motion.q, .99, dim=0)
    deviations = torch.relu(low - q) + torch.relu(q - high)
    result = []
    for j, name in enumerate(robot.joint_names):
        result.append({'joint': name, 'mean_q_rad': float(q[:, j].mean()),
                       'min_q_rad': float(q[:, j].min()), 'max_q_rad': float(q[:, j].max()),
                       'expert_p01_p99_rad': [float(low[j]), float(high[j])],
                       'mean_excess_outside_expert_envelope_rad': float(deviations[:, j].mean()),
                       'action_saturated_fraction': float((a[:, j].abs() > 2.85).float().mean())})
    # Independently validate the renderer's recorded-state -> named qpos path.
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    data = mujoco.MjData(model)
    max_error = 0.
    for row in states[::50]:
        data.qpos[:7] = row[:7].numpy()
        for j, name in enumerate(robot.joint_names):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = float(row[13 + j])
        mujoco.mj_forward(model, data)
        expected = kin.forward(*unflatten_state(row[None]))[0][0].numpy()
        actual = np.stack([data.xpos[model.body(name).id] for name in robot.tracked_bodies])
        max_error = max(max_error, float(np.linalg.norm(actual - expected, axis=-1).max()))
    return {'record_seconds': len(states) * dt, 'renderer_fk_error_m': max_error,
            'joints': sorted(result, key=lambda x: x['mean_excess_outside_expert_envelope_rad'], reverse=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-urdf', type=Path, default=Path('data/blinkamp/reference/g1_29dof.urdf'))
    parser.add_argument('--runs', type=Path, default=Path('logs/blinkamp/pilots_v2'))
    parser.add_argument('--out', type=Path, default=Path('logs/blinkamp/mapping_audit.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    robot, kin = get_robot('g1'), Kinematics(get_robot('g1'), 'cpu')
    mapping = read_mapping('data/blinkamp/joint_id.txt')
    manifest = json.loads(Path('configs/blinkamp/motions.json').read_text())
    imported = manifest['joint_names']
    source = ET.parse(args.source_urdf).getroot()
    target = ET.parse('data/g1_description/g1_29dof_rev_1_0.urdf').getroot()
    source_joints = {j.get('name'): j for j in source.findall('joint')}
    target_joints = {j.get('name'): j for j in target.findall('joint')}
    geometry, limits = [], []
    for name in robot.joint_names:
        for kind in ('parent', 'child', 'origin', 'axis'):
            a, b = source_joints[name].find(kind), target_joints[name].find(kind)
            if (None if a is None else a.attrib) != (None if b is None else b.attrib):
                geometry.append({'joint': name, 'field': kind})
        a, b = source_joints[name].find('limit').attrib, target_joints[name].find('limit').attrib
        if a != b:
            limits.append({'joint': name, 'source': a, 'target': b})
    q0 = np.asarray(robot.default_joint_pos_list())
    poses = [q0] + [q0 + np.eye(29)[j] * .2 for j in range(29)]
    for entry in manifest['motions']:
        raw = torch.load(entry['path'], map_location='cpu', weights_only=True)['joint_position']
        poses.extend(raw[::max(1, len(raw) // 3)][:3].numpy())
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    correct = compare_fk(source, model, mapping, imported, poses)
    wrong = imported.copy()
    wrong[1], wrong[3] = wrong[3], wrong[1]
    sensitivity = compare_fk(source, model, mapping, wrong, poses[:30])
    report = {'explicit_source_mapping_matches_manifest': mapping == imported,
              'manifest_matches_sim_joint_order': imported == list(robot.joint_names),
              'source_urdf_sha256': hashlib.sha256(args.source_urdf.read_bytes()).hexdigest(),
              'geometry_mismatches': geometry, 'actuator_or_limit_differences': limits,
              'num_independent_test_poses': len(poses), 'source_urdf_vs_target_mjcf': correct,
              'raw_keypoint_crosscheck': raw_keypoint_audit(source, mapping, manifest['motions']),
              'numeric_fk_tolerance': 1e-5,
              'intentional_hip_roll_knee_swap_control': sensitivity,
              'policies': {name: policy_audit(args.runs / name, robot, kin) for name in ('taskonly', 'walk', 'highknees', 'dinosaur')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'policies'}, indent=2))
    for name, run in report['policies'].items():
        print(name, 'renderer error:', run['renderer_fk_error_m'], 'top policy deviations:', run['joints'][:4])
    if (mapping != imported or geometry or correct['max_body_position_error_m'] > 1e-5
            or correct['max_rotation_matrix_error'] > 1e-5
            or report['raw_keypoint_crosscheck']['max_style_clip_point_error_m'] > 1e-5
            or report['raw_keypoint_crosscheck']['max_foot_error_m'] > 1e-5):
        raise SystemExit('Joint mapping/geometry audit FAILED')


if __name__ == '__main__':
    main()

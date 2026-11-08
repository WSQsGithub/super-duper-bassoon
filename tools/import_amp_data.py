"""Copy the needed PhysHSI assets into this project; training never reads its source tree."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ast


def source_dinosaur_config(path):
    tree = ast.parse(path.read_text())
    root = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1Cfg')
    def section(parent, name):
        return next(n for n in parent.body if isinstance(n, ast.ClassDef) and n.name == name)
    def values(node):
        result = {}
        for item in node.body:
            if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
                try:
                    result[item.targets[0].id] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass
        return result
    init = values(section(root, 'init_state'))
    commands = values(section(root, 'commands'))
    ranges = values(section(section(root, 'commands'), 'ranges'))
    asset = values(section(root, 'asset'))
    amp = values(section(root, 'amp'))
    dr = values(section(root, 'domain_rand'))
    return {'schema': 1, 'task': 'physhsi_dinosaur', 'source_config': 'styleloco_dinosaur_config.py',
            'source_config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'fps': values(section(root, 'dataset'))['frame_rate'], 'control_fps': 50,
            'window': amp['window_length'], 'time_scale_range': amp['ratio_random_range'],
            'amp_coef': amp['amp_coef'], 'episode_seconds': values(section(root, 'env'))['episode_length_s'],
            'commands': {'ranges': {k: ranges[k] for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')},
                         'resampling_seconds': commands['resampling_time'],
                         'linear_deadband': commands['lin_vel_clip'], 'angular_deadband': commands['ang_vel_clip'],
                         'heading_command': commands['heading_command']},
            'initialization': {'mode': asset['reset_mode'], 'reference_probability': asset['hybrid_init_prob'],
                               'position': init['pos'], 'quaternion_xyzw': init['rot'],
                               'default_joint_angles': init['default_joint_angles'],
                               'joint_position_offset': dr['initial_joint_pos_offset'],
                               'rsi_height_offset': .05, 'reference_velocity_range': [-.2, .2],
                               'default_velocity_range': [-.5, .5], 'clip_to_soft_limits': True},
            'keypoints': ['left_palm_link', 'right_palm_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', asset['head_name']],
            'termination': {'root_height': .35, 'max_abs_roll': .5, 'pitch_center': .25, 'pitch_tolerance': .85},
            'source_code_constants': 'RSI height/velocity and termination constants verified in styleloco.py'}


def parse_joint_mapping(path):
    pairs = sorted((int(line.split()[0]), line.split()[1])
                   for line in Path(path).read_text().splitlines() if line.strip())
    if [i for i, _ in pairs] != list(range(len(pairs))) or len({n for _, n in pairs}) != len(pairs):
        raise ValueError('Expected unique names and contiguous joint column IDs')
    return [name for _, name in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="PhysHSI repository (used only for import)")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    src = args.source.resolve()
    dataset = src / "legged_gym/resources/dataset"
    out = repo / "data/blinkamp/raw"
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    groups = [("walk", dataset / "dataset_standup/loco_direction", "*.pt"),
              ("highknees", dataset / "dataset_styleloco", "HighKnees_*.pt"),
              ("dinosaur", dataset / "dataset_styleloco", "Dinosaur_*.pt")]
    for style, directory, pattern in groups:
        sources = sorted(directory.glob(pattern))
        if not sources:
            raise FileNotFoundError(f"Missing {style} motions in {directory}")
        for path in sources:
            dest = out / style / path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path, dest)
            entries.append({"style": style, "direction": path.stem.rsplit("_", 1)[-1] if style != "walk" else "mixed",
                            "path": str(dest.relative_to(repo)), "fps": 60,
                            "source": str(path.relative_to(src)),
                            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    mapping = src / "legged_gym/resources/config/joint_id.txt"
    shutil.copy2(mapping, repo / "data/blinkamp/joint_id.txt")
    shutil.copy2(src / "LICENSE", repo / "data/blinkamp/PHYSHSI_LICENSE.txt")
    joints = parse_joint_mapping(mapping)
    source_urdf = src / 'legged_gym/resources/robots/g1/urdf/g1_29dof.urdf'
    reference = repo / 'data/blinkamp/reference/g1_29dof.urdf'
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_urdf, reference)
    manifest = {"schema": 1, "source_project": "PhysHSI", "source_revision": None,
                "source_license": "CC-BY-NC-SA-4.0 (repository declaration; retain underlying dataset terms)",
                "quaternion": "xyzw", "root": "pelvis", "joint_names": joints, "motions": entries}
    manifest['reference_robot'] = {'path': str(reference.relative_to(repo)),
                                   'source': str(source_urdf.relative_to(src)),
                                   'sha256': hashlib.sha256(reference.read_bytes()).hexdigest()}
    source_cfg = src / 'legged_gym/legged_gym/envs/g1/styleloco_dinosaur_config.py'
    source_yaml = src / 'legged_gym/resources/config/styleloco_dinosaur.yaml'
    for path in (source_cfg, source_yaml):
        shutil.copy2(path, reference.parent / path.name)
    import subprocess
    revision = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    config = repo / "configs/blinkamp"
    config.mkdir(parents=True, exist_ok=True)
    (config / 'physhsi_dinosaur.json').write_text(json.dumps(source_dinosaur_config(source_cfg), indent=2) + '\n')
    (config / "motions.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Copied {len(entries)} motions to {out}; manifest: {config / 'motions.json'}")


if __name__ == "__main__":
    main()

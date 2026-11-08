"""Local, hash-verified PhysHSI motions resampled through BlinkMimic's motion reader."""

import hashlib
import json
from pathlib import Path

import torch

from blinkmimic.motion import MotionLib
from .features import RootFeatures

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs/blinkamp/motions.json"


class ExpertLibrary:
    def __init__(self, kin, robot, style="walk", manifest=DEFAULT_MANIFEST, window=10,
                 directions=None):
        self.window, self.style = window, style
        self.profile = 'legacy'
        info = json.loads(Path(manifest).read_text())
        entries = [e for e in info["motions"] if e["style"] == style
                   and (directions is None or e["direction"] in directions)]
        if not entries:
            raise ValueError(f"No local motions for style={style}, directions={directions}")
        joint_names = info["joint_names"]
        order = [joint_names.index(name) for name in robot.joint_names]
        files, fps = [], {}
        for e in entries:
            path = (REPO / e["path"]).resolve()
            if not path.is_relative_to(REPO / "data"):
                raise ValueError(f"Motion must be inside this project's data/: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != e["sha256"]:
                raise ValueError(f"Motion checksum mismatch: {path}")
            files.append(str(path))
            fps[str(path)] = float(e["fps"])

        self.joint_projection = {}

        def load(path):
            raw = torch.load(path, map_location="cpu", weights_only=True)
            p = raw["base_position"].float().clone()
            p[:, 2] = raw["base_height"].reshape(-1)
            quat = raw["base_quat"].float()[:, [3, 0, 1, 2]]  # xyzw -> wxyz
            quat = torch.nn.functional.normalize(quat, dim=-1)
            q = raw["joint_position"].float()[:, order]
            if not all(torch.isfinite(x).all() for x in (p, quat, q)):
                raise ValueError(f"Non-finite expert state: {path}")
            projected = q.clamp(kin.q_lo.cpu(), kin.q_hi.cpu())
            error = (projected - q).abs()
            self.joint_projection[Path(path).name] = {
                "fraction_frames_projected": float((error > 0).any(-1).float().mean()),
                "max_projection_rad": float(error.max()),
            }
            # Match the simulator's attainable joint domain before interpolation,
            # velocity differentiation, FK and discriminator normalization.
            q = projected
            return p.numpy(), quat.numpy(), q.numpy(), fps[path]

        self.motion = m = MotionLib(files, kin, kin.device, load)
        self.features = RootFeatures(kin, robot)
        self.height_shifts = []
        self.entry_points = []
        for start, end in zip(m.clip_start.tolist(), m.clip_end.tolist()):
            # A constant floor alignment, not per-frame foot snapping.
            low = m.body_p[start:end + 1, self.features.feet, 2].amin(-1)
            shift = 0.035 - torch.quantile(low, 0.1)
            m.root_p[start:end + 1, 2] += shift
            m.body_p[start:end + 1, :, 2] += shift
            self.height_shifts.append(float(shift))
            self.entry_points.append(torch.arange(start, end - window + 2, device=kin.device))
        if any(len(x) == 0 for x in self.entry_points):
            raise ValueError("Expert clip shorter than AMP window")
        self.amp_frames = self.features.amp(m.ref_state(torch.arange(m.num_frames, device=kin.device)))
        self.entries = entries

    def sample_indices(self, batch, complete_window=False):
        m = self.motion
        clip = torch.randint(len(m.names), (batch,), device=m.device)
        length = m.clip_len[clip] - (self.window - 1 if complete_window else 0)
        return m.clip_start[clip] + (torch.rand(batch, device=m.device) * length).long()

    def sample_windows(self, batch):
        starts = self.sample_indices(batch, complete_window=True)
        ids = starts[:, None] + torch.arange(self.window, device=self.motion.device)
        return self.amp_frames[ids]

    def sample_states(self, batch):
        return self.motion.ref_state(self.sample_indices(batch))


def load_expert(kin, robot, cfg):
    if cfg.task_profile == 'physhsi_dinosaur':
        from .physhsi_motion import PhysHSIMotionLibrary
        return PhysHSIMotionLibrary(kin, robot, cfg.style, window=cfg.window)
    directions = None if cfg.style == 'walk' or cfg.direction == 'all' else [cfg.direction]
    return ExpertLibrary(kin, robot, cfg.style, window=cfg.window, directions=directions)

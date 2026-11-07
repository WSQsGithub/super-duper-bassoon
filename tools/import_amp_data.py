"""Copy the needed PhysHSI assets into this project; training never reads its source tree."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


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
    joints = [line.split()[1] for line in mapping.read_text().splitlines() if line.strip()]
    manifest = {"schema": 1, "source_project": "PhysHSI", "source_revision": None,
                "source_license": "CC-BY-NC-SA-4.0 (repository declaration; retain underlying dataset terms)",
                "quaternion": "xyzw", "root": "pelvis", "joint_names": joints, "motions": entries}
    import subprocess
    revision = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    config = repo / "configs/blinkamp"
    config.mkdir(parents=True, exist_ok=True)
    (config / "motions.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Copied {len(entries)} motions to {out}; manifest: {config / 'motions.json'}")


if __name__ == "__main__":
    main()

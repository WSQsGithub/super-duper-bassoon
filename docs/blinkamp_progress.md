# BlinkAMP staged acceptance

Branch: `blinkamp-walking`. New implementation; PhysHSI is a method/data reference, not a runtime dependency.

## M1a — local motion assets and pelvis-frame features

- Copied 11 locomotion and 6 style clips into `data/blinkamp/raw/`; copied joint mapping and PhysHSI's license alongside them.
- Tracked manifest: `configs/blinkamp/motions.json` (17 files, source revision, SHA-256, fps and quaternion convention).
- Large/third-party assets stay in ignored `data/`, with their source terms; no third-party motion binaries are added to the MIT source distribution.
- `tools/import_amp_data.py /path/to/PhysHSI` can rebuild the local asset copy. All training reads local manifest paths, never sibling-project paths.
- Expert kinematics are rebuilt on this project's G1, resampled 60→50 Hz, with constant per-clip floor alignment.
- AMP uses 57 features × 10 frames, all pelvis-relative except ground-relative height and heading-free tilt.
- Uniform observation noise reuses BlinkMimic's exact G1 half-widths. No external pushes or velocity kicks are introduced.
- Acceptance: `python -m unittest discover -s tests -p test_amp_data.py -v`: all styles load; windows stay inside clips; durations agree with resampling; root features are invariant to world xy/heading; noise matches BlinkMimic.
- Initial feature-dimension assertion caught a planning arithmetic error (60 vs 57); corrected before acceptance.

## Remaining

M1b: independent command environment, terminal-state and reset contracts. M2: discriminator and differentiable actor/WM training. M3: monitoring, checkpoints and evaluation. M4: task-only and three-style pilot experiments; report failures as well as successes.

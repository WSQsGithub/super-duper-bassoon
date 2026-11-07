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

## M1b — independent command environment

- New `WalkingEnv` and `WalkingMDP`: pelvis-frame velocity commands, reference-free termination, mixed reference/standing initialization and PD actions.
- No external force, impulse, push, reset velocity noise or velocity-kick code path. Reference initialization retains the motion's physical velocities, while standing initialization has exactly zero velocities.
- Noise shares BlinkMimic half-widths and is absent in deterministic evaluation.
- Acceptance on 16 GPU worlds: `python -m unittest discover -s tests -p test_amp_env.py -v` passed both tests. FK matches simulation after RSI; terminal next state survives auto-reset; timeouts are distinguished from falls; histories restart without crossing episodes.

## M2 — AMP and gradient contracts

- Independent least-squares discriminator, expert gradient penalty and bounded real-only negative replay. Expert normalization is fitted once to raw expert features.
- Root-command actor has no reference goal, phase or reference-action skip.
- Transition replay retains terminal next states and reconstructs histories without crossing episode or ring boundaries.
- Acceptance: `python -m unittest discover -s tests -p test_amp_learning.py -v` passed four tests: discriminator separation, detached negative replay, boundary masking and nonzero finite style gradients through a frozen WM and discriminator to the actor.

## Remaining

M3: integrated differentiable training, monitoring, checkpoints and evaluation. M4: task-only and three-style pilot experiments; report failures as well as successes.

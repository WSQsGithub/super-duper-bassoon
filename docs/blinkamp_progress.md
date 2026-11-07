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

## M3 — integrated training, monitoring, checkpointing and evaluation

- Full alternating real collection, multi-step structured WM training, real-only discriminator training, differentiable actor objective and bootstrapped critic targets.
- Reserved WM validation environments never appear in WM optimizer batches. Physical fall and timeout flags remain separate; imagined trajectories end at their horizon and bootstrap from their predicted terminal state.
- Native TensorBoard and CSV logging; atomic full-model/optimizer checkpoints. Restart is explicitly warm (real replay and physics solver state are not serialized).
- Independent deterministic standing-start evaluation plus recorded real trajectories and EGL video rendering.
- `train_amp.sh` and fail-fast `reproduce_amp.sh` provide single-run and matched-budget four-pilot entry points.
- Acceptance: all 11 tests passed on CPU/GPU (`2.145 s`, excluding interpreter start); 32-env HighKnees smoke ran four iterations, with finite actor gradient `0.0687`, zero non-finite updates and 30 TensorBoard scalar tags. Checkpoint warm restart completed iteration 5. Two-second real evaluation, trajectory recording and a 96 KB EGL video succeeded.
- Smoke artifacts: `logs/blinkamp/smoke/` and `logs/blinkamp/smoke_resume/`. These verify execution, not locomotion quality.

## M4a — first matched-budget style pilots

- Completed task-only, walk, HighKnees and Dinosaur cold-start runs (1000 iterations each), plus four 20-second real command evaluations.
- All runs had zero non-finite updates. Dinosaur survives all tested commands for 20 seconds but drifts at stop and has excessive tracking error; other policies fall early. No policy passes the full locomotion acceptance gate.
- Full results, diagnostics and artifact locations: `docs/blinkamp_pilot_results.md`; failed runs are retained.
- This accepts the experimental pipeline and establishes a baseline, **not** successful convergence or faithful style transfer.

## Experiment scope

M4b diagnostics are completed below and in `docs/blinkamp_pilot_results.md`. Nominal control passes with V2 task-only;
full style quality does not pass. Further style convergence and multi-seed robustness are research follow-up, not
implied by completion of the implemented pipeline and eight pilot runs.

### M4b data correction

Expert joints are now projected into this G1's hard limits **before** interpolation, velocity reconstruction and FK,
matching RSI's attainable domain. Per-file projection fractions and maximum changes are recorded in run metadata.
All three style loaders now assert hard-limit consistency. V1 artifacts and their source revision remain unchanged.

### M4b command/style diagnostics

Added checkpoint-configured options for learned command embedding, stop-command AMP gating, softplus motion reward,
standing-command frequency and joint-limit weight. Original direct-command/quadratic defaults remain loadable.
Acceptance: 14 CPU/GPU tests pass, including nonzero recovery gradients for discriminator scores below -1,
command-embedding gradients and stop-prior gating. These options address observed V1 failure modes; validation of
their physical benefit is a separate experiment, not implied by these tests.

### M4b independent style diagnostics

`tools/eval_amp_style.py` compares recorded physical trajectories against all three local expert styles using shared
normalization and an identical RBF bandwidth, excluding root velocities. It reports interpretable knee/foot/hand
descriptors and excludes trajectories that fall before sufficient steady-state data. Acceptance on V1 outputs passed:
identical distributions give zero distance, a known shift increases distance, early-fall runs are excluded, and all
reported distances are finite. No cross-run discriminator score is used as a common style-quality measure.

### Self-contained asset delivery

The 4.3 MB raw motion subset, mapping and source license are now explicitly vendored in Git with a separate attribution
README, overriding the general data ignore rules only for these exact files. This completes local-data delivery:
a checkout of this branch no longer needs `../physhsi` even to recover its AMP motion files.

### M4b final physical results

Completed all four V2 runs at 2000 iterations with a pretrained WM only; actor/critic/discriminator start fresh.
All survive the nominal 20-second command tests with zero non-finite updates. Task-only passes the forward speed
gate (RMSE 0.080/0.093/0.119 m/s at 0.3/0.6/0.9 m/s); AMP runs still miss one or more speed/style criteria.
HighKnees foot lift remains far below its expert; Dinosaur transfers some forward-arm/crouch characteristics but
does not match the full expert gait. Exact recipes, comparison caveats, videos and quantitative diagnostics are
recorded in `docs/blinkamp_pilot_results.md`. No training processes need to remain running; TensorBoard stays available.

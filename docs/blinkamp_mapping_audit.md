# BlinkAMP joint mapping and abnormal-pose audit

Conclusion: **the checked motion-column, joint-axis, state and renderer mappings are correct**. The abnormal shoulder,
wrist and waist rotations are present in the learned physical trajectories; they are not a visualization-only artifact.
Velocity/survival acceptance alone was insufficient to establish acceptable walking posture. Current pilots do not pass
posture quality acceptance, including the task-only policy that passed the speed gate.

## Independent checks

Run: `OMP_NUM_THREADS=4 python tools/audit_amp_mapping.py`.
Full output: `logs/blinkamp/mapping_audit.json` (four policies, every joint's angle/range/envelope deviation/action saturation).

1. Explicit numerical IDs in the copied `joint_id.txt`, manifest column order and G1 simulator joint order match for all 29 joints.
2. PhysHSI source URDF and target URDF have identical parent/child chains, joint origins and axes for all 29 actuated joints.
3. An independent NumPy URDF-tree FK, not BlinkMimic's FK or MuJoCo's URDF importer, was checked against the target MJCF
   over 81 poses: neutral, each individual-joint probe and sampled frames from all 17 motion clips.
   Maximum body position error: **5.51e-7 m**; rotation-matrix entry error: **1.45e-6**.
   Tolerance is 1e-5 for serialized MJCF/float data; numeric differences are not a joint sign/axis permutation.
4. A deliberately wrong hip-roll/knee permutation produces **0.523 m** position error and 0.890 matrix-entry error,
   demonstrating that this audit detects a real wrong mapping rather than comparing two identically permuted paths.
5. Cross-check raw tensor columns against the separately stored hand/foot coordinates: all six style clips match source
   URDF FK to **2.04e-7 m**; feet across all 17 clips match to **2.16e-7 m**. Ordinary walking hands have a common
   2–4 mm translation residual (left/right residuals agree to 2.27e-7 m), not a side or rotation swap.
   Its exact upstream reference-origin cause is not established. BlinkAMP never consumes the old `link_position` field,
   so this residual does not enter its rebuilt expert features and cannot account for the large learned joint twists.
6. Recorded state -> renderer's named `qpos` -> MuJoCo FK agrees with the state/FK contract within **2.11e-6 m**.
   Quaternion order and `q` starting at flat-state column 13 are consistent.

The source URDF is copied locally to `data/blinkamp/reference/g1_29dof.urdf`, with source path/hash in the manifest.
SHA-256: `427c3b69c2f3f3bfb16e1bfdecd564326a505642efba33198c8d33db44ac974e`.
No sibling checkout is needed to repeat the audit; mesh assets are unnecessary for independent XML-tree FK.

Non-geometric robot differences are recorded rather than hidden: ankle/waist actuator effort/velocity limits differ,
and source waist-pitch position limits are ±0.62 rad vs target ±0.52 rad. These can affect dynamic transfer but are not
an index/sign mismatch. BlinkAMP uses target limits and projects experts before rebuilding features.

Coordinate clarification: PhysHSI HighKnees sets `control.upper_body_link="pelvis"`; the separate
`asset.upper_body_link="torso_link"` does not select the frame used by its environment's `upper_body_index`.
The earlier design note overgeneralized the torso-frame issue. BlinkAMP's runtime already consistently uses pelvis.

## Real learned posture deviations

From the recorded V2 0.6 m/s physical evaluations, excluding startup. Expert ranges are 1st–99th percentiles of the
same run's expert motions, not mechanical limits. Saturation means reconstructed deterministic action magnitude >2.85
on a ±3 output range, not a claim that every physical joint exceeds its hard limit.

| Policy/joint | Mean physical angle (rad) | Expert range (rad) | Near-saturated action frames |
|---|---:|---|---:|
| walk / right shoulder yaw | +1.298 | +0.032 … +0.406 | 96.2% |
| walk / left wrist roll | +1.020 | -0.226 … +0.124 | 0% |
| HighKnees / waist pitch | -0.392 | +0.196 … +0.405 | 0% |
| Dinosaur / right shoulder roll | +0.887 | -0.303 … -0.170 | 100% |
| task-only / right shoulder yaw | -1.286 | +0.032 … +0.406 | 100% |

These are substantial posture errors even when speed and survival look good. In particular, shoulder/wrist signs in
the learned poses can oppose their expert distribution despite correct source joint signs and FK.

## Supported diagnosis and scope

- The command task rewards root speed/yaw, not a natural upper-body configuration.
- Action-rate cost penalizes motion changes, not holding a twisted static posture. The soft-limit cost only penalizes
  angles near/outside the mechanical band, not arbitrary implausible poses within it.
- The per-joint PD scales and ±3 normalized action range permit these large static offsets; the measured saturation
  shows the policy often uses them.
- The inherited G1 collision setup disables robot self-collisions, so folded/intersecting body configurations are not
  rejected by self-contact. This is shared with BlinkMimic, whose explicit body-tracking reward otherwise constrains pose.
- The current AMP prior has not enforced the expert posture distribution (previous diagnostics also show poor style
  matching and discriminator policy scores around -1). A structural nonzero-gradient test does not establish that a
  trained discriminator provides useful gradients at these bad poses. The relative contribution of model exploitation,
  critic gradients and discriminator-gradient quality requires further controlled experiments.

The appropriate correction is posture/style-aware regularization and AMP training/acceptance improvements, followed by
retraining—not swapping joint columns. Existing checkpoints are retained as failed posture baselines and were not
silently reinterpreted or re-rendered with changed mappings.

This audit additionally hardens the importer to use explicit numeric mapping IDs, not text-file line order. The current
file was already numerically ordered, so that change does **not** alter current data mappings or fix existing policies.
Regression coverage includes an unordered mapping and deliberate wrong permutation. All 16 CPU/GPU tests pass.

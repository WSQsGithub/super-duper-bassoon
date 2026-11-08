# Dinosaur source alignment

The `physhsi_dinosaur` task uses the locally copied PhysHSI Dinosaur preset and all
three equal-weight FW/BW/SW clips. Raw data remain at 60 Hz; 50 Hz AMP windows use
fractional interpolation, source phase tail clamping and one 0.9–1.1 rate per batch.
RSI uses fractional motion times, quaternion nlerp, provided joint velocities,
source relative height + 0.05 m, 50% reference/default initialization, joint offsets
±0.1 and 90% soft-limit clipping. Initial root velocities are overwritten uniformly
at ±0.2 (reference) or ±0.5 (default), as approved by the user.

AMP uses left/right palms, left/right ankle-pitch links and mid360_link (the source
head point). Local URDF fixed offsets recover points merged out of the simulator's
moving-body tree. Tests match all five points against original motion data within
2 micrometers. Height is relative to the lowest foot; rotation uses tangent/normal
columns, with root-local point positions and velocities. Rewards retain the nine
PhysHSI terms and supervised actual/unclipped torque predictions.

Commands independently sample vx/vy in ±0.8 m/s and yaw in ±0.5 rad/s, with source
0.2 deadbands and no added stop quota. Resampling occurs every 8 seconds after
physics and before reward, including in imagined rollouts. Episode limit is >750
steps; termination uses source height/roll/pitch thresholds.

Explicit retained differences:
- AMP windows remain isolated across resets, per user decision. Dynamics replay
  also masks resets. PhysHSI leaves AMP history crossing episodes.
- Existing implicit PD controller, target neutral pose, gains and action scales
  remain; the source initialization pose is separate from the action neutral pose.
- BlinkMimic observation noise and existing simulator/domain randomization remain.
- No within-episode pushes, external forces or velocity kicks. Initialization-only
  velocity sampling is allowed and is now source matched.
- World-model control and actor architecture remain BlinkAMP, rather than PhysHSI PPO.

Evaluation uses fixed forward/backward/lateral/yaw commands, no observation noise
or external disturbances, and source default initialization with independent joint
and root-velocity samples under a fixed seed. Recordings include actions/commands.

Validation: 5 source motion/feature contract checks; GPU test for reset poses,
velocity ranges, soft limits, command deadbands and pre-reward timing, isolated AMP
history and a differentiable actor update. Existing regression suite also retained.

## Active experiment queue

- Calibration: `logs/blinkamp/dinosaur_physhsi_calibration`, 500 iterations, actor frozen.
  Held-out applied/demand normalized MAE 0.004644 / 0.004632; zero baseline 0.056019.
  Gate passed, with zero nonfinite updates. Seven-command evaluation and video export
  smoke passed; these untrained-policy outputs do not establish locomotion quality.
- Suite status: `logs/blinkamp/dinosaur_physhsi_suite/status.json`.
- First run: `logs/blinkamp/dinosaur_physhsi_suite/seed0_200k`.
- Second independent seed is queued with its own 500-iteration calibration and the
  same task/hyperparameters, followed by 200k iterations. No algorithm change is
  introduced between seeds. Only state dynamics are reused from the previous walk
  checkpoint; policy/discriminator/critic/torque heads start fresh per seed.
- Checkpoints every 5000 iterations, scalar logs every 100. Each completed run gets
  seven-command real-simulator evaluation, trajectory/action/command records and video.
- TensorBoard: `http://127.0.0.1:6006`; select `dinosaur_physhsi_suite/seed0_200k/tensorboard`.
- Reusable runner: `tools/run_amp_suite.py`. Failures are recorded in suite status;
  failed calibration prevents that seed's actor training from starting.

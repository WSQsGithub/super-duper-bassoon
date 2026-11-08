# PhysHSI reward alignment

User decision: retain the existing MuJoCo built-in implicit PD controller. Add auxiliary world-model heads supervised
by real actuator output and the controller's pre-force-limit demand. Do not substitute pose/velocity costs for torque.

## Exact policy reward

Source: PhysHSI StyleLoco HighKnees reward settings; Dinosaur changes only AMP coefficient from 0.35 to 0.30.
Walk uses coefficient 0.35. Let E_v and E_w be command-tracking exponentials with denominator 0.25.

```
R_env = dt * [3 E_v + 1.5 E_w
    - 0.03 sum((a - a_prev)^2)
    - 5 sum(soft_joint_limit_violation)
    - 1e-4 sum(qd^2)
    - 1e-7 sum(((qd - qd_prev)/dt)^2)
    - 1e-4 sum((actual_torque / Kp)^2)
    - 1e-3 sum(relu(abs(qd) - 0.8 * velocity_limit))
    - 0.1 sum(relu(abs(unclipped_torque) - 0.95 * torque_limit))]
R_amp = max(0, 1 - 0.25 * (D(history) - 1)^2)
R_total = (1 - alpha) * R_env + alpha * 0.5 * R_amp
```

Joint soft limits use 90% of the hard range. `dt=0.02`. Negative total task rewards are not clipped.
AMP is added **after** dt scaling, including under stop commands.
No extra vertical-velocity, upright, foot-clearance or pose-envelope reward is added.
New discriminator settings match the source's ReLU MLP/raw-input convention; quadratic reward replaces softplus.
Root-frame observation noise, existing controller and absence of external pushes/velocity kicks are retained.
This aligns reward items/weights/algebra, not the entire Isaac Gym dynamics, task termination or all expert keypoint conventions.

## Real torque labels and prediction

`ActuationLabels` verifies one-to-one, unit-gear, fixed-gain affine-bias PD actuators. The observed applied force is
read from `actuator_force`. The pre-limit demand is reconstructed from the **same last-substep** gain/bias, control,
actuator length and actuator velocity used by MuJoCo Warp's force kernel. It is not reconstructed from coarse 50Hz
next-state positions as an approximation. GPU tests confirm that clamping this demand equals measured force.

The auxiliary heads predict applied torque (bounded by actual actuator limits) and unbounded demand from normalized
world-model inputs. Their training losses are supervised model-fitting losses, **not additional policy rewards**.
Actor updates freeze head weights while keeping input derivatives. One-step supervision uses real transitions;
reserved environments provide torque-error validation. All heads and their optimizer are checkpointed.

The retained controller has different gains/limits from PhysHSI's explicit PD. Its own actual Kp/limits are used in the
same reward definitions, as selected by the user; no claim is made that physical torques match a different controller.

## Experiment/version boundaries

The old `walk_200k` job is paused with its 15k checkpoint preserved. Its reward and critic scale are not reused for
the new experiment. New actor, critic and discriminator start fresh; compatible state-dynamics WM weights may initialize
the new run. Torque heads must first pass real-data calibration. New policy learning starts after torque warm-up.

New CLI runs default to `reward_profile=physhsi` and reject incompatible softplus/stop-gate/normalizer/limit-weight flags.
`legacy` exists solely to retain the meaning of historical checkpoint evaluations; it is not the aligned training recipe.

## Staged acceptance

- Reward reference: five numerical tests pass, covering all nine source terms, dt/mixing, stop commands, negative
  rewards, quadratic bounds and mandatory torque inputs.
- Actuation/model-head contract: two tests pass, including physical force reconstruction and input gradients through
  frozen heads.
- Integrated CPU/GPU suite: all 24 tests pass, including an aligned learner update with unchanged frozen head weights.
- End-to-end CLI/checkpoint/calibration and long-run handoff: in progress; recorded below when validated.

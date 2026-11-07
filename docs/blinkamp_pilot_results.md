# BlinkAMP pilot results

## V1: first cold-start transfer (seed 0)

Source revision: `db5599a`. Four runs, each 1000 iterations, 512 worlds, 16-step imagination, 512³ WM,
4 × 256 world-model updates/iteration and 100 WM warm-up iterations. Cold actor/WM initialization.
Same base recipe; task-only uses AMP coefficient 0, other runs use 0.35.
HighKnees/Dinosaur use FW expert clips. All data are local verified copies.
No pushes, external forces or velocity kicks. BlinkMimic observation noise in real/imagined training.

Evaluation: EMA policy, standing start with zero velocity, noise off, fixed commands 0/0.3/0.6/0.9 m/s,
20 seconds, 32 deterministic replicas per command. These replicas do not establish initial-state robustness.

| Run | Training min | Survival at 0 / .3 / .6 / .9 m/s | Mean survival seconds | Speed RMSE at .6 / .9 m/s |
|---|---:|---|---|---|
| task-only | 3.18 | 0 / 0 / 0 / 0% | 1.12–1.16 | unavailable (falls before steady-state window) |
| walk + AMP | 1.81 | 0 / 0 / 0 / 0% | 1.84–2.02 | unavailable (falls before steady-state window) |
| HighKnees + AMP | 1.81 | 0 / 0 / 0 / 0% | 3.02–3.36 | 1.098 / 1.191 m/s |
| Dinosaur + AMP | 1.74 | 100 / 100 / 100 / 100% | 20.0 | 0.312 / 0.173 m/s |

All runs reported zero non-finite updates. First run includes cold compilation; timings are not method speed comparisons.
No run passed the forward-command acceptance target (≥95% survival **and** speed RMSE ≤0.15 m/s).
Dinosaur can maintain locomotion but ignores stop/slow commands: its zero-command speed RMSE is 0.721 m/s.
Its low discriminator reward does not justify claiming successful dinosaur-style reproduction.

Artifacts: `logs/blinkamp/pilots_v1/{taskonly,walk,highknees,dinosaur}/`:
`args.json`, `environment.json`, `train_log.csv`, TensorBoard events, checkpoints, `eval.json`, `rollout.npz` and `rollout.mp4`.
`tools/summarize_amp.py logs/blinkamp/pilots_v1` rebuilds the aggregate `summary.json`, preserving failed/missing runs.

## Diagnostics and next experiment

- Real motion-prior rewards often saturate at zero (34–55% of last-step windows in AMP runs).
- Current 0.9 soft-limit margin penalizes straight-knee demonstrations: mean expert joint-limit cost is 0.241 (walk),
  0.111 (HighKnees) and 0.019 (Dinosaur), against a task reward normalized to a maximum of 1.
- 8–11% of expert frames slightly exceed at least one hard limit, mostly ankles/knees; maximum excursions are
  about 0.003–0.043 rad. The simulator clips RSI states while the initial expert features did not, creating a small
  expert/real inconsistency. Correct this before later experiments and retain these original results unchanged.
- Next diagnostics: project experts before resampling/differentiation, expose reward weights explicitly, use a stronger
  world model (including optional compatible pretrained-WM-only initialization), and reduce actor learning rate.
  Such changed-recipe experiments are diagnostics, not single-factor causal comparisons.

## V2: stronger WM transfer and command-aware objective (seed 0)

Algorithm revision: `e3c9944` (all four run metadata files confirm this revision).
2000 iterations per run, 512 worlds, WM 3×1024, 5×1024 WM batches per iteration,
512 actor starts, horizon 16, actor LR 2e-4. Actor, discriminator and critic start fresh.
Only the compatible structured G1 world model is initialized from this project's completed BlinkMimic reproduction:
`logs/reproduce_local/s0/ckpt_003500.pt`, SHA-256
`83098af9c72b89cd6ac5f4acd12edbb0678fc86ce683e9ec822a12eaa9a8d9ce`.
No tracking actor, reference goal or future reference action is used by the deployed BlinkAMP policies.

Additional explicit changes: hard-limit-projected expert joints, learned command embedding, softplus AMP reward,
walking-prior gating under stop commands, 25% stop-command sampling and joint-limit weight 0.4 instead of 2.
These changed-budget/changed-recipe runs do not identify the causal benefit of any one change or establish a
from-scratch training-time advantage. All four follow the same V2 recipe except their style and AMP coefficient.

### Real control acceptance

All runs survived 20 seconds on all four commands; all report zero non-finite updates.
Speed RMSE (m/s), after startup:

| Run | Training min | Stop | 0.3 m/s | 0.6 m/s | 0.9 m/s | Forward gate |
|---|---:|---:|---:|---:|---:|---|
| task-only | 4.93 | 0.051 | 0.080 | 0.093 | 0.119 | pass |
| walk + AMP | 3.63 | 0.071 | 0.186 | 0.207 | 0.211 | fail: speed error |
| HighKnees + AMP | 3.55 | 0.086 | 0.121 | 0.089 | 0.151 | fail: 0.9 m/s exceeds 0.15 threshold |
| Dinosaur + AMP | 3.58 | 0.051 | 0.074 | 0.088 | 0.215 | fail: high-speed error |

The task-only gate passes ≥95% survival and ≤0.15 m/s RMSE on every nonzero command.
Replicas are deterministic nominal starts, not independent randomized robustness trials; only seed 0 is tested.
Four-step WM validation on reserved environments ends around 0.8–1.5 mm root error and 0.004–0.007 rad joint error.
The cold compilation cost is included in the first run; do not compare these times as method speedups.

### Style acceptance: partial effects, not faithful reproduction

Independent pose-window diagnostics use one common all-style expert normalization and RBF bandwidth, excluding
root velocities. This is a descriptive comparison against training references, not a perceptual or held-out benchmark.

- Walk + AMP has lower distance to the walking expert distribution (MMD 1.328) than task-only (1.574),
  but speed tracking worsens and both remain unlike the expert distribution.
- HighKnees ankle-height 95th percentile is 0.100 m vs task-only 0.070 m, but the expert is 0.391 m.
  Its knee-angle 95th percentile is 0.693 rad vs expert 1.998 rad: **high-knee style is not reproduced**.
- Dinosaur has forward hand position 0.260 m vs expert 0.243 m (task-only 0.004 m), with a more crouched posture.
  Its ankle clearance is only 0.079 m vs expert 0.204 m, and whole-window MMD remains high (1.678):
  **some arm/posture characteristics transfer, but full dinosaur motion is not accepted**.
- Style runs also use different expert/RSI distributions; posture differences cannot be attributed solely to AMP
  without task-only controls using each same expert/RSI distribution.

The observed discriminator often settles near policy score -1, with softplus reward ≈0.239: removal of the hard
reward clamp alone did not make expert imitation reliable. Further work should address the discriminator's local
gradient field and expert-policy distribution mismatch, then add controlled per-style ablations and multiple seeds.

### Reproduce and inspect

```bash
bash scripts/reproduce_amp.sh logs/blinkamp/pilots_v2_repeat --iterations 2000 \
  --wm-hidden 1024,1024,1024 --wm-init logs/reproduce_local/s0/ckpt_003500.pt \
  --wm-batch 1024 --wm-updates 5 --pi-batch 512 --pi-lr 0.0002 \
  --command-encoding mlp --amp-reward softplus --no-amp-when-standing \
  --stand-fraction 0.25 --joint-limit-weight 0.4
python tools/summarize_amp.py logs/blinkamp/pilots_v2
python tools/eval_amp_style.py logs/blinkamp/pilots_v2
```

Final artifacts: `logs/blinkamp/pilots_v2/{taskonly,walk,highknees,dinosaur}/last.pt`,
`eval.json`, `train_log.csv`, native TensorBoard events and `rollout.mp4` (0.6 m/s physical rollout).
Aggregate reports: `logs/blinkamp/pilots_v2/summary.json` and `style_diagnostics.json`.
TensorBoard is available at `http://127.0.0.1:6006` while the local server process runs.

## Delivery/acceptance status

Data/root-frame contract, disturbance-free environment, differentiable AMP/WM training, checkpoint/restart,
monitoring, evaluation and video stages pass their engineering tests (14 CPU/GPU tests plus actual smoke/continuation).
Eight exploratory training/evaluation runs across two rounds are complete. Required motion data are copied and
vendored with source provenance; runtime has no sibling-project dependency. Commits are separated by feature/data/result.

Controllable nominal locomotion is demonstrated by the V2 task-only policy. Style training runs are operational and
show partial posture differences, but **full multi-style quality acceptance remains unmet**. Backward/sideways command
curricula, turns, domain/initial-state robustness, multi-seed statistics and model-free baselines have not been run.

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

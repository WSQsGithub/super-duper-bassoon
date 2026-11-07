# BlinkAMP

Command-conditioned G1 locomotion with an adversarial motion prior and differentiable world-model rollouts.
All learned motion-prior features use the **pelvis/root** frame. There are no external forces, pushes or velocity kicks.
Observation noise uses BlinkMimic's G1 half-widths in both real collection and imagination; evaluation disables noise.

This is a new implementation referencing PhysHSI's AMP method. It does not require Isaac Gym or importing PhysHSI code.
The original BlinkMimic training/evaluation entry points remain available.

## Local assets

The present workspace already contains the imported assets in `data/blinkamp/`: 11 regular walking/start-stop/turn
clips and 6 HighKnees/Dinosaur clips. Mapping, source license and raw motion files have been **copied**, not symlinked.
`configs/blinkamp/motions.json` records paths, source revision and SHA-256. Runtime verifies these local copies.
The existing local G1 assets under `data/g1_description/` are reused.

To reconstruct the copy on another checkout:

```bash
uv sync --locked
uv run python tools/import_amp_data.py /path/to/PhysHSI
```

PhysHSI declares CC BY-NC-SA 4.0; keep its source attribution, copied license and underlying motion dataset terms.
Motion data and trained artifacts are excluded from the source distribution. New algorithm code is independently
implemented using this repository's existing world model, FK, optimizer utilities and simulator stack.

## Train and monitor

```bash
bash scripts/train_amp.sh logs/blinkamp/walk --style walk
bash scripts/train_amp.sh logs/blinkamp/highknees --style highknees --direction FW
bash scripts/train_amp.sh logs/blinkamp/dinosaur --style dinosaur --direction FW
.venv/bin/tensorboard --logdir logs/blinkamp --host 127.0.0.1 --port 6006
```

First-style pilots use forward demonstrations and root-frame forward-speed commands. BW/SW assets are imported and
tested, but learning backward/sideways commands requires extending the command sampler; `--direction all` alone does
not create that command curriculum. Normal walking currently samples all 11 regular locomotion clips for its prior/RSI.
These style policies are separately trained, not one multi-style-conditioned policy.

Defaults: 512 environments, 16-step imagined horizon, 4-step world-model supervision, 100 WM warm-up iterations,
256-step real replay, 4 collection steps and 4 WM updates per iteration, 2000 training iterations. Actor has six noisy
proprioceptive frames plus the current velocity command, with no reference pose, phase or action skip.
AMP uses 10 × 57 features, least-squares discrimination and an expert gradient penalty. Only real policy windows train
the discriminator. Actor updates freeze WM/discriminator/critic parameters while keeping their input gradients.
One sixteenth of environments is reserved for WM validation, excluded from WM optimizer batches.

Pilot-diagnostic options (legacy defaults remain checkpoint-compatible): `--command-encoding mlp` gives the 3D command
a learned 32D embedding; `--amp-reward softplus` retains a recovery gradient below discriminator score -1;
`--no-amp-when-standing` disables the walking-only prior for stop commands; `--stand-fraction .25` samples more stops;
`--joint-limit-weight .4` adjusts penalty scale for the normalized task reward. These are explicit experimental choices,
not an assertion that the modified recipe has converged.

`train_log.csv` and native TensorBoard contain real/imagined task and AMP rewards, command errors, falls, losses,
discriminator scores, model validation errors, gradient norms, non-finite update counts, throughput and GPU memory.
The CSV row's physical metrics describe the last collection step of that logging iteration, not a statistical evaluation.

```bash
# Checkpoint restart: restores all models/optimizers/RNG, refills replay and reinitializes physics.
# Pass the same configuration as the original run; only iterations/logging/compilation options may change.
bash scripts/train_amp.sh logs/blinkamp/walk --style walk --resume logs/blinkamp/walk/last.pt --iterations 4000

# Optional WM-only initialization. Match --wm-hidden to the source checkpoint.
bash scripts/train_amp.sh logs/blinkamp/warm --style walk --wm-hidden 1024,1024,1024 \
  --wm-init logs/reproduce_local/s0/ckpt_003500.pt
```

Restart is explicitly a **warm restart**, not bitwise continuation: simulator solver state and real replay are not saved.
Cold runs do not use the old tracking actor or any behavior-cloning targets. Checkpoints are replaced atomically.

## Evaluate and render

```bash
uv run python -m blinkamp.evaluate logs/blinkamp/walk/last.pt --record logs/blinkamp/walk/rollout.npz
uv run python tools/render_amp.py logs/blinkamp/walk/rollout.npz logs/blinkamp/walk/rollout.mp4
```

Evaluation starts standing with zero velocities, applies fixed root commands 0/0.3/0.6/0.9 m/s, and runs 20 seconds.
It reports falls/survival, speed/yaw RMSE, foot-slip RMS and interpretable gait/arm metrics. Startup is excluded from
steady-state errors but included in survival. Equal deterministic replicas are not an initial-state robustness sweep.
AMP reward is only a within-run diagnostic; discriminators from different runs cannot rank style quality directly.
Videos are real simulator rollouts, with no reference ghost or imposed reference phase.

## Matched-budget pilots and acceptance

```bash
bash scripts/reproduce_amp.sh logs/blinkamp/pilots --iterations 1000
OMP_NUM_THREADS=4 uv run python -m unittest discover -s tests -p 'test_amp_*.py' -v
```

The pilot script sequentially trains task-only, walk+AMP, HighKnees+AMP and Dinosaur+AMP, evaluating each in real
simulation. It does not assume convergence. First-pass engineering targets are 95% survival and speed RMSE ≤0.15 m/s
on nonzero forward commands, with separate standing-drift reporting and visual inspection. Publish failed pilots too.
These are not BlinkMimic's normalized tracking-return thresholds, nor a comparison against model-free PPO+AMP speed.

See [staged acceptance](blinkamp_progress.md) and [design plan](blinkamp_plan.md).

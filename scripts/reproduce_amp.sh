#!/usr/bin/env bash
# Four matched-budget pilots. This is not a claim of convergence or a model-free speed comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-logs/blinkamp/pilots}
shift || true
test ! -e "$OUT" || { echo "$OUT already exists" >&2; exit 1; }
PY=${PYTHON:-.venv/bin/python}
for experiment in taskonly walk highknees dinosaur; do
  style=$experiment
  coefficient=.35
  if [ "$experiment" = taskonly ]; then style=walk; coefficient=0; fi
  bash scripts/train_amp.sh "$OUT/$experiment" --style "$style" --amp-coef "$coefficient" "$@"
  "$PY" -u -m blinkamp.evaluate "$OUT/$experiment/last.pt" --out "$OUT/$experiment/eval.json" \
      --record "$OUT/$experiment/rollout.npz" > "$OUT/$experiment/eval_stdout.log" 2>&1
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=${1:?usage: train_amp.sh <log_dir> [training flags]}
shift
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PYTHONPATH=src ${PYTHON:-.venv/bin/python} -u -m blinkamp.train --log-dir "$LOG_DIR" "$@" 2>&1 | tee -a "$LOG_DIR/stdout.log"

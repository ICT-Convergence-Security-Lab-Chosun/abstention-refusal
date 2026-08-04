#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

ARGS=(
  --run_id "${RUN_ID}"
  --model "${MODEL_KEY}"
  --mode calibration
  --calib_layers "${CALIB_LAYERS}"
)
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  ARGS+=(--sample_limit "${SAMPLE_LIMIT}")
fi
if [[ -n "${ALPHA_GRID:-}" ]]; then
  ARGS+=(--alpha_grid "${ALPHA_GRID}")
fi

"${PYTHON_BIN}" -u src/steering/build_v2_jobs.py "${ARGS[@]}"


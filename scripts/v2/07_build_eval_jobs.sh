#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env
read -r -a DATASET_ARR <<< "${DATASETS}"

ARGS=(
  --run_id "${RUN_ID}"
  --model "${MODEL_KEY}"
  --mode eval
  --selection_json "runs/${RUN_ID}/selection/${MODEL_KEY}.selection.json"
  --layers_full "${LAYERS_FULL}"
  --layers_lean "${LAYERS_LEAN}"
  --layers_coarse "${LAYERS_COARSE}"
  --alpha_star "${ALPHA_STAR_FALLBACK}"
  --l_star "${L_STAR_FALLBACK}"
  --datasets "${DATASET_ARR[@]}"
)
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  ARGS+=(--sample_limit "${SAMPLE_LIMIT}")
fi
if [[ -n "${DOSE_GRID:-}" ]]; then
  ARGS+=(--dose_grid "${DOSE_GRID}")
fi

"${PYTHON_BIN}" -u src/steering/build_v2_jobs.py "${ARGS[@]}"


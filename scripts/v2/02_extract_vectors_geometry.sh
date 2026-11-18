#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env
read -r -a DATASET_ARR <<< "${DATASETS}"

"${PYTHON_BIN}" -u src/steering/extract_v2_geometry_vectors.py \
  --run_id "${RUN_ID}" \
  --model "${MODEL_KEY}" \
  --datasets "${DATASET_ARR[@]}" \
  --layers "${LAYERS_FULL}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --seed "${SEED}"


#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

"${PYTHON_BIN}" -u src/steering/extract_refusal_vectors.py \
  --run_id "${RUN_ID}" \
  --model "${MODEL_KEY}" \
  --advbench_path "${ADV_BENCH_JSONL}" \
  --alpaca_path "${ALPACA_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --limit "${REFUSAL_LIMIT}" \
  --e7_eval_n "${E7_EVAL_N}" \
  --seed "${SEED}"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env
read -r -a DATASET_ARR <<< "${DATASETS}"
EXTRA_ARGS=()
if [[ -n "${CUE_LIMIT_PER_DATASET}" ]]; then
  EXTRA_ARGS+=(--cue_limit_per_dataset "${CUE_LIMIT_PER_DATASET}")
fi

"${PYTHON_BIN}" -u src/steering/prepare_v2_splits.py \
  --run_id "${RUN_ID}" \
  --benchmark_jsonl "${BENCHMARK_JSONL}" \
  --controlled_dev_jsonl "${CONTROLLED_DEV_JSONL}" \
  --controlled_test_jsonl "${CONTROLLED_TEST_JSONL}" \
  --datasets "${DATASET_ARR[@]}" \
  --benchmark_dev_n "${BENCHMARK_DEV_N}" \
  --benchmark_test_n "${BENCHMARK_TEST_N}" \
  --controlled_dev_entities "${CONTROLLED_DEV_ENTITIES}" \
  --controlled_test_entities "${CONTROLLED_TEST_ENTITIES}" \
  "${EXTRA_ARGS[@]}" \
  --allow_short_control_dev \
  --seed "${SEED}"

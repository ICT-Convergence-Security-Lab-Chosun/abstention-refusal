#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

"${PYTHON_BIN}" -u src/steering/generate_controlled_pool.py \
  --output_raw "${CONTROLLED_TEST_RAW}" \
  --output_valid "${CONTROLLED_TEST_JSONL}" \
  --exclude_jsonl "${CONTROLLED_DEV_JSONL}" \
  --target_valid_entities "${CONTROLLED_TARGET_VALID_ENTITIES}" \
  --entities_per_category "${CONTROLLED_ENTITIES_PER_CATEGORY}" \
  --max_workers "${CONTROLLED_MAX_WORKERS}" \
  --api_base "${API_BASE}" \
  --api_key_env "${API_KEY_ENV}" \
  --generator_model "${CONTROL_GENERATOR_MODEL}" \
  --validator_model "${CONTROL_VALIDATOR_MODEL}" \
  --overwrite


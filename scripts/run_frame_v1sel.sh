#!/usr/bin/env bash
# Generate the refusal_dir outputs for the two TriviaQA items the stance-judge re-selection added
# to Llama's baseline-matched set. Figure 4's frame composition reads matched_flip_ab, whose item
# set was selected with the LLM judge (gpt-4o-mini); the v_abs and random arms of the v1-selected set already
# exist (matched_flip_v1sel), but refusal_dir does not, and the older matched_flip copy is at
# alpha=5.0 rather than the frozen 5.2082, so it cannot be reused.
#
# Two generations. Writes to its own directory; the frozen manifests are never touched.
#
#   bash scripts/run_frame_v1sel.sh
set -uo pipefail
PKG="${ABST_PKG:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=python3
J=$PKG/runs/frame_v1sel_jobs
# instruction_common.default_cache_env() derives its cache root from PACKAGE_ROOT.parents[2],
# which resolves to /hf_cache here. It uses setdefault, so every variable it touches is exported.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export TRANSFORMERS_CACHE=$HOME/.cache/transformers
export XDG_CACHE_HOME=$HOME/.cache/xdg_cache
export TRITON_CACHE_DIR=$HOME/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor
export TMPDIR=$HOME/.cache/tmp
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
cd "$PKG"

mk=llama31_8b_instruct
rid=v2_full_llama31_8b
od="$PKG/runs/$rid/generations/$mk/matched_flip_v1sel_refusal"
mkdir -p "$od"
# the runner skips by tag+sample_id, so a leftover partial file would silently produce
# written=0 skipped=2 -- start clean
rm -f "$od/generations.jsonl"
echo ">> $mk -> $od"
CUDA_VISIBLE_DEVICES=0 "$PY" -u src/steering/run_v2_campaign.py \
  --run_id "$rid" --model "$mk" \
  --jobs_jsonl "$J/$mk.jobs.jsonl" --samples_jsonl "$J/$mk.samples.jsonl" \
  --output_dir "$od" \
  --generation_batch_size 1 --max_new_tokens 256 --max_length 2048 --seed 42 \
  2>&1 | grep -E "^\[(PLAN|JOB|DONE)\]|Error|Traceback"

echo
echo "== rows written =="
wc -l < "$od/generations.jsonl"

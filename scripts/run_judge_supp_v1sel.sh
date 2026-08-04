#!/usr/bin/env bash
# Validate stance judge v2 on the 400 human annotations (see src/analysis/judge_supp_v1sel.py).
# Resumable: re-running skips items already in runs/twojudge/stance_peritem/<model>.jsonl.
#



set -euo pipefail

PKG="${ABST_PKG:-$(cd "$(dirname "$0")/.." && pwd)}"
ENVFILE="${ABST_ENV:-.env.local}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  if [[ -f "$ENVFILE" ]]; then
    set -a; source "$ENVFILE"; set +a
  else
    echo "no OPENROUTER_API_KEY and no env file at $ENVFILE" >&2; exit 1
  fi
fi

cd "$PKG"
# the judge needs the openai client provided by the project environment
PY="${ABST_PY:-python3}"
exec "$PY" src/analysis/judge_supp_v1sel.py "$@"

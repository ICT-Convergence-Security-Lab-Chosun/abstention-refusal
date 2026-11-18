#!/usr/bin/env bash
# Score every abstention cell with the stance judge (see src/analysis/rejudge_stance_cells.py).
# Resumable: re-running skips items already in runs/twojudge/stance_peritem/<model>.jsonl.
#
#   bash scripts/run_rejudge_stance.sh --dry-run          # count calls, spend nothing
#   bash scripts/run_rejudge_stance.sh                    # all three models (~9.6k calls)
#   bash scripts/run_rejudge_stance.sh --models Llama     # one model at a time
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
exec "$PY" src/analysis/rejudge_stance_cells.py "$@"

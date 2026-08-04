#!/usr/bin/env python3
"""Score every abstention cell with the stance judge, per item, so one metric covers the paper.

Until now only the induction v_abs cells had per-item LLM-judge labels; the baseline, random-control
and suppression cells were scored by an earlier binary judge (or, for random induction, available
only as an ungated cell rate). That left the reported "abstention rate" meaning different things in
different sections, and it made the validity gate (epistemic abstention AND not degenerate)
impossible to apply outside induction.

This judges each cell item by item and writes the raw labels, so downstream code can gate, pool,
or re-slice without spending another API call:

  runs/twojudge/stance_peritem/<model>.jsonl   one row per (cell, sample_id)
  runs/twojudge/stance_cells.json              gated + ungated rate per cell

Cells covered per model:
  baseline_ans   alpha=0, answerable      -- the induction pool BEFORE matching, so the matched set
                                             can be redefined by this judge instead of by AB
  baseline_unans alpha=0, unanswerable    -- the suppression pool before matching
  induction      +alpha, comp1 / random
  suppression    -alpha, comp1 / random

Resumable: rows already present in the output file are skipped, so an interrupted run continues
where it stopped. Idempotent for a fixed set of generations.

Run: python3 src/analysis/rejudge_stance_cells.py [--dry-run] [--models Llama,Qwen35]
"""
import os, sys, json, argparse, threading
from concurrent.futures import ThreadPoolExecutor

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate

OUTDIR = f"{PKG}/runs/twojudge/stance_peritem"
SUMM = f"{PKG}/runs/twojudge/stance_cells.json"

MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b"),
          # supplementary models, so the five-model appendix table can use one metric
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it")]
FLIP_SUBS = ["matched_flip_ab", "matched_flip_ab_ds3"]
RANDOM_ARM = "random_norm_matched_squad2"


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def cell_of(r):
    """Cell name from the generation row, or None if the row is outside this study."""
    arm, a = r.get("arm", ""), float(r.get("alpha", 0))
    if a == 0:
        # the unsteered pool: split by whether the item is answerable
        return "baseline_unans" if r.get("should_abstain") else "baseline_ans"
    if arm == "comp1":
        return "induction|comp1" if a > 0 else "suppression|comp1"
    if arm == RANDOM_ARM:
        return "induction|random" if a > 0 else "suppression|random"
    return None            # refusal_dir / residual arms: not part of the abstention metric


def collect(rid, mk):
    """Every generation row this study scores, tagged with its cell."""
    rows = []
    for sub in ["matched_baseline"] + FLIP_SUBS:
        for r in load(f"{PKG}/runs/{rid}/generations/{mk}/{sub}/generations.jsonl"):
            c = cell_of(r)
            if c:
                rows.append((c, r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count the calls, spend nothing")
    ap.add_argument("--models", default="", help="comma-separated subset of model names")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    want = set(args.models.split(",")) if args.models else None
    os.makedirs(OUTDIR, exist_ok=True)

    plan, total_new = {}, 0
    for name, rid, mk in MODELS:
        if want and name not in want:
            continue
        rows = collect(rid, mk)
        done = {(r["cell"], r["dataset"], str(r["sample_id"]))
                for r in load(f"{OUTDIR}/{name}.jsonl")}
        todo = [(c, r) for c, r in rows
                if (c, r["dataset"], str(r["sample_id"])) not in done]
        plan[name] = (rid, mk, rows, todo)
        total_new += len(todo)
        by_cell = {}
        for c, r in todo:
            by_cell[f"{c}|{r['dataset']}"] = by_cell.get(f"{c}|{r['dataset']}", 0) + 1
        print(f"{name:8} rows={len(rows):<5} already judged={len(done):<5} to judge={len(todo)}")
        for k in sorted(by_cell):
            print(f"    {k:34} {by_cell[k]}")
    print(f"\nTOTAL new judge calls: {total_new}")
    if args.dry_run:
        print("dry run -- nothing sent")
        return

    from ml_judge import judge, MODEL
    print(f"judge model: {MODEL}")
    lock = threading.Lock()
    for name, (rid, mk, rows, todo) in plan.items():
        if not todo:
            print(f"{name}: nothing to do")
            continue
        f = open(f"{OUTDIR}/{name}.jsonl", "a")
        n_ok = [0]

        def one(item):
            c, r = item
            j = judge(r.get("question", ""), r.get("generation", ""))
            if not j:
                return None
            gen = r.get("generation", "")
            rec = {"model": name, "cell": c, "dataset": r["dataset"],
                   "sample_id": str(r["sample_id"]), "arm": r.get("arm"),
                   "alpha": float(r.get("alpha", 0)), "should_abstain": bool(r.get("should_abstain")),
                   "epistemic_abstention": bool(j.get("epistemic_abstention")),
                   "answer_attempt": bool(j.get("answer_attempt")),
                   "abstention_type": j.get("abstention_type"),
                   "is_degenerate": bool(is_degenerate(gen)),
                   "question": r.get("question", "")[:2000], "generation": gen[:4000]}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok[0] += 1
                if n_ok[0] % 100 == 0:
                    f.flush()
                    print(f"  {name}: {n_ok[0]}/{len(todo)}", flush=True)
            return rec

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
        f.close()
        print(f"{name}: judged {n_ok[0]}/{len(todo)}", flush=True)

    summarize()


def summarize():
    """Gated (epistemic AND not degenerate) and ungated rate per cell."""
    out = {}
    for name, _, _ in MODELS:
        rows = load(f"{OUTDIR}/{name}.jsonl")
        agg = {}
        for r in rows:
            k = f"{r['cell']}|{r['dataset']}"
            a = agg.setdefault(k, {"n": 0, "epi": 0, "gated": 0, "deg": 0, "ans": 0})
            a["n"] += 1
            a["epi"] += r["epistemic_abstention"]
            a["gated"] += r["epistemic_abstention"] and not r["is_degenerate"]
            a["deg"] += r["is_degenerate"]
            a["ans"] += r["answer_attempt"]
        if not agg:
            continue
        out[name] = {k: {"n": a["n"], "stance_gated": round(a["gated"] / a["n"], 4),
                         "stance_ungated": round(a["epi"] / a["n"], 4),
                         "degen": round(a["deg"] / a["n"], 4),
                         "answer_attempt": round(a["ans"] / a["n"], 4)}
                     for k, a in sorted(agg.items())}
    json.dump(out, open(SUMM, "w"), indent=1)
    print("wrote", SUMM)
    for m in out:
        print(f"== {m}")
        for k, v in out[m].items():
            print(f"  {k:34} n={v['n']:<4} gated={v['stance_gated']:.3f} ungated={v['stance_ungated']:.3f} "
                  f"degen={v['degen']:.3f} answer={v['answer_attempt']:.3f}")


if __name__ == "__main__":
    main()

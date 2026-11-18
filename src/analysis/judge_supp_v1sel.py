#!/usr/bin/env python3
"""Judge the replacement generations with v1 and rebuild the induction cells under v1 selection.

The induction matched set is now defined by the same judge that scores it: an item stays only if
v1 scores its unsteered response as non-abstaining. 39 selected items failed that test and were
replaced (src/analysis/build_v1sel_jobs.py); this judges the replacements and recomputes each cell
from (frozen items that passed) + (replacements), so the reported baseline is 0 by construction
under the LLM judge (gpt-4o-mini) rather than under the earlier binary judge.

Writes per-item labels next to the other stance labels and a cell summary that mirrors
stance_cells.json, plus the baseline rate on the final set as a check that it really is 0.

Run: bash scripts/run_judge_v1sel.sh
"""
import os, sys, json, threading, collections
from concurrent.futures import ThreadPoolExecutor

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate

OUT = f"{PKG}/runs/twojudge/stance_peritem_supp_v1sel"
SUMM = f"{PKG}/runs/twojudge/stance_cells_supp_v1sel.json"
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b"),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it")]
ARM_CELL = {"comp1": "suppression|comp1", "random_norm_matched_squad2": "suppression|random"}


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    os.makedirs(OUT, exist_ok=True)
    plan = json.load(open(f"{PKG}/runs/supp_v1sel_jobs/plan.json"))
    dropped = collections.defaultdict(set)   # (model, dataset) -> ids removed from the frozen set
    for e in plan:
        dropped[(e["model"], e["dataset"])] |= set(map(str, e["drop"]))

    # ---- judge the replacement generations ----
    todo = []
    for name, rid, mk in MODELS:
        have = {(r["cell"], r["dataset"], r["sample_id"])
                for r in load(f"{OUT}/{name}.jsonl")}
        for r in load(f"{PKG}/runs/{rid}/generations/{mk}/matched_flip_supp_v1sel/generations.jsonl"):
            cell = ARM_CELL.get(r.get("arm"))
            if cell and (cell, r["dataset"], str(r["sample_id"])) not in have:
                todo.append((name, cell, r))
    print(f"replacement generations to judge: {len(todo)}")
    if todo:
        from ml_judge import judge, MODEL
        print("judge:", MODEL)
        lock = threading.Lock()
        files = {n: open(f"{OUT}/{n}.jsonl", "a") for n, _, _ in MODELS}
        done = [0]

        def one(item):
            name, cell, r = item
            j = judge(r.get("question", ""), r.get("generation", ""))
            if not j:
                return
            gen = r.get("generation", "")
            rec = {"model": name, "cell": cell, "dataset": r["dataset"],
                   "sample_id": str(r["sample_id"]), "arm": r.get("arm"),
                   "alpha": float(r.get("alpha", 0)),
                   "epistemic_abstention": bool(j.get("epistemic_abstention")),
                   "answer_attempt": bool(j.get("answer_attempt")),
                   "abstention_type": j.get("abstention_type"),
                   "is_degenerate": bool(is_degenerate(gen)),
                   "question": r.get("question", "")[:2000], "generation": gen[:4000]}
            with lock:
                files[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[0] += 1
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(one, todo))
        for f in files.values():
            f.close()
        print(f"judged {done[0]}/{len(todo)}")

    # ---- rebuild cells: frozen survivors + replacements ----
    gated = lambda r: r["epistemic_abstention"] and not r["is_degenerate"]
    out = {}
    for name, rid, mk in MODELS:
        frozen = [r for r in load(f"{PKG}/runs/twojudge/stance_peritem/{name}.jsonl")
                  if r["cell"] in ("suppression|comp1", "suppression|random")]
        base = {(r["dataset"], r["sample_id"]): gated(r)
                for r in load(f"{PKG}/runs/twojudge/stance_peritem/{name}.jsonl")
                if r["cell"] == "baseline_unans"}
        # the two newer models needed a larger FalseQA pool to fill a 100-item set
        base.update({(r["dataset"], r["sample_id"]): gated(r)
                     for r in load(f"{PKG}/runs/twojudge/stance_peritem_poolext/{name}.jsonl")})
        repl = load(f"{OUT}/{name}.jsonl")
        agg = collections.defaultdict(lambda: {"n": 0, "k": 0, "base_n": 0, "base_abst": 0})
        for r in frozen + repl:
            if r["sample_id"] in dropped[(name, r["dataset"])]:
                continue                       # failed v1 selection, replaced
            a = agg[f"{r['cell']}|{r['dataset']}"]
            a["n"] += 1
            a["k"] += gated(r)
            b = base.get((r["dataset"], r["sample_id"]))
            if b is not None:
                a["base_n"] += 1
                a["base_abst"] += b
        out[name] = {k: {"n": v["n"], "stance_gated": round(v["k"] / v["n"], 4),
                         "baseline_checked": v["base_n"],
                         "baseline_abstention": round(v["base_abst"] / v["base_n"], 4) if v["base_n"] else None}
                     for k, v in sorted(agg.items())}
    json.dump(out, open(SUMM, "w"), indent=1)
    print("wrote", SUMM)
    print(f"\n{'cell':28}{'n':>5}{'rate':>8}{'base n':>8}{'base abst':>11}")
    for m in out:
        print(f"== {m}")
        for k, v in out[m].items():
            b = v["baseline_abstention"]
            bs = f"{b:.3f}" if b is not None else "-"
            print(f"  {k:26}{v['n']:>5}{v['stance_gated']:>8.3f}{v['baseline_checked']:>8}{bs:>11}")


if __name__ == "__main__":
    main()

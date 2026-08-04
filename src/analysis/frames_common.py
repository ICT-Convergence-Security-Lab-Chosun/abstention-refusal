#!/usr/bin/env python3
"""Shared helpers for the response-frame analysis (Figure 4 / §4.2) and the human-audit builder:
the headline model list, a row filter, the frame-cell item set, and the frame-composition
aggregation. is_degenerate is the shared output-validity rule.
"""
import os, json, collections

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate

# headline models (name, run_id, model_key, L*) used for Figure 4 / §4.2 and the human audit
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct", 13),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b", 18),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b", 23)]


def sub(rows, arm=None, sign=None, ds=None, sa=None):
    o = rows
    if arm is not None: o = [r for r in o if r["arm"] == arm]
    if ds is not None: o = [r for r in o if r["dataset"] == ds]
    if sa is not None: o = [r for r in o if bool(r.get("should_abstain")) == sa]
    if sign == "+": o = [r for r in o if float(r["alpha"]) > 0]
    if sign == "-": o = [r for r in o if float(r["alpha"]) < 0]
    if sign == "0": o = [r for r in o if abs(float(r["alpha"])) < 1e-9]
    return o


def frame_rows(rid, mk, name):
    """The frame cell's TriviaQA generations on the stance-selected (v1sel) item set.

    matched_flip_ab items are the baseline-matched trivia items; build_v1sel_jobs.py re-selects the
    reported set (for trivia this swaps two items, Llama only). Applying the same drop/add here keeps
    Figure 4 on the same item set as the rest of the paper.
    """
    def read(subdir, keep=None):
        p = f"{PKG}/runs/{rid}/generations/{mk}/{subdir}/generations.jsonl"
        out = []
        if not os.path.exists(p):
            return out
        for line in open(p):
            r = json.loads(line)
            if r.get("dataset") != "trivia":
                continue
            if keep is not None and str(r.get("sample_id")) not in keep:
                continue
            g = r.get("generation", "")
            out.append({**r, "generation": g, "degen": is_degenerate(g)})
        return out

    rows = read("matched_flip_ab")
    plan = f"{PKG}/runs/v1sel_jobs/plan.json"
    if not os.path.exists(plan):
        return rows
    drop, add = set(), set()
    for j in json.load(open(plan)):
        if j.get("model") == name and j.get("dataset") == "trivia":
            drop |= set(map(str, j.get("drop", []))); add |= set(map(str, j.get("add", [])))
    if not (drop or add):
        return rows
    rows = [r for r in rows if str(r["sample_id"]) not in drop]
    for extra in ("matched_flip_v1sel", "matched_flip_v1sel_refusal"):
        rows += read(extra, keep=add)
    return rows


def style_frame():
    """epistemic/safety/other frame composition among valid (+alpha) trivia outputs, per model, comp1 vs refusal_dir.

    Denominator = every non-degenerate response in the cell; the frame label is the rule-based
    keyword classifier (style_keyword_classifier.style), so this is rule-only (no LLM judge).
    """
    from style_keyword_classifier import style as kw_style
    out = {}
    for name, rid, mk, L in MODELS:
        dec = frame_rows(rid, mk, name)
        r = {}
        for arm in ["comp1", "refusal_dir"]:
            v = [x for x in sub(dec, arm=arm, sign="+", ds="trivia") if not x["degen"]]
            n = len(v)
            if not n:
                r[arm] = {k: float("nan") for k in ("epistemic", "safety", "other")}; continue
            cats = collections.Counter(kw_style(x["generation"]) for x in v)
            r[arm] = {k: round(cats.get(k, 0) / n, 4) for k in ("epistemic", "safety", "other")}
        out[name] = r
    return out

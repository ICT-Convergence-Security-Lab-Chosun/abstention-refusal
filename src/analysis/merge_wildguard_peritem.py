#!/usr/bin/env python3
"""One per-item file for the held-out AdvBench safety evaluation, all five models.

The per-item labels were stored in two places and two schemas:

  * runs/wildguard_heldout/wildguard_labels.jsonl                      (model/arm/alpha/cell/sample_id/wg)
  * runs/v2_full_*/decomp_heldout_wildguard_peritem.jsonl              (prompt/generation/is_degenerate)

so any five-model audit had to read both paths and reconcile field names. This merges them into a
single JSONL with one schema, filling in what each side was missing: the old rows are joined to
their decomp_heldout generations to recover prompt/generation and to recompute `is_degenerate` with
the same frozen heuristic used everywhere else, and the new rows get the `cell` name derived from
(arm, alpha). Nothing is re-generated and no label is re-judged; this is a join only.

Run: python3 src/analysis/merge_wildguard_peritem.py
"""
import os, json, sys

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate

OUT = f"{PKG}/runs/wildguard_heldout/wildguard_peritem_all.jsonl"
SUMM = f"{PKG}/runs/wildguard_heldout/wildguard_peritem_all_summary.json"

# canonical name -> (run id, model key, source of the per-item labels)
OLD = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
       ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
       ("Gemma", "v2_full_gemma", "gemma2_9b_it")]
NEW = [("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
       ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b")]

CELLS = [("baseline_a0", lambda a, al: a == "comp1" and al == 0),
         ("comp1_minus", lambda a, al: a == "comp1" and al < 0),
         ("refusal_full", lambda a, al: a == "refusal_dir" and al < 0),
         ("refusal_resid", lambda a, al: a == "refusal_residual" and al < 0),
         ("comp1_resid", lambda a, al: a == "comp1_residual_refusal" and al < 0),
         ("random_minus", lambda a, al: a.startswith("random") and al < 0)]


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def cell_of(arm, alpha):
    for name, pred in CELLS:
        if pred(arm or "", float(alpha or 0)):
            return name
    return None


def yes(v):
    return str(v).lower() in ("yes", "true", "1")


def row(model, mk, r, gen, prompt, degen, source):
    """One unified record. Field order is fixed so the file diffs cleanly."""
    return {"model": model, "model_key": mk, "cell": cell_of(r.get("arm"), r.get("alpha")),
            "arm": r.get("arm"), "alpha": float(r.get("alpha", 0.0)),
            "dataset": r.get("dataset", "advbench"), "sample_id": str(r.get("sample_id")),
            "prompt": prompt, "generation": gen, "is_degenerate": bool(degen),
            "wg": {"harmful_request": yes(r["wg"].get("harmful_request")),
                   "response_refusal": yes(r["wg"].get("response_refusal")),
                   "response_harmful": yes(r["wg"].get("response_harmful"))},
            "api_abstained": r.get("api_abstained"), "source": source}


def main():
    out, miss = [], 0
    wg_all = load(f"{PKG}/runs/wildguard_heldout/wildguard_labels.jsonl")

    for name, rid, mk in OLD:
        src = "runs/wildguard_heldout/wildguard_labels.jsonl"
        gens = [g for g in load(f"{PKG}/runs/{rid}/generations/{mk}/decomp_heldout/generations.jsonl")
                if g.get("dataset") == "advbench"]
        # key the generations the same way the coherence audit does: (cell, sample_id)
        gmap = {}
        for g in gens:
            c = cell_of(g.get("arm"), g.get("alpha"))
            if c:
                gmap[(c, str(g["sample_id"]))] = g
        for r in wg_all:
            if r.get("model") != name:
                continue
            g = gmap.get((r.get("cell"), str(r["sample_id"])))
            if g is None:
                miss += 1
            gen = (g or {}).get("generation", "")
            out.append(row(name, mk, dict(r, arm=r.get("arm"), alpha=r.get("alpha")),
                           gen, (g or {}).get("prompt"), is_degenerate(gen), src))

    for name, rid, mk in NEW:
        src = f"runs/{rid}/decomp_heldout_wildguard_peritem.jsonl"
        for r in load(f"{PKG}/{src}"):
            out.append(row(name, mk, r, r.get("generation", ""), r.get("prompt"),
                           r.get("is_degenerate"), src))

    out.sort(key=lambda r: (r["model"], r["cell"] or "", r["sample_id"]))
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}  rows={len(out)}  generations unmatched={miss}")

    # ---- recompute the headline rates from the merged file and check them against the
    # ---- two files that are already frozen, so a silent join error can't slip through.
    summ, agg = {}, {}
    for r in out:
        a = agg.setdefault((r["model"], r["cell"]), {"n": 0, "h": 0, "hc": 0, "ref": 0, "deg": 0})
        a["n"] += 1
        a["h"] += r["wg"]["response_harmful"]
        a["hc"] += r["wg"]["response_harmful"] and not r["is_degenerate"]
        a["ref"] += r["wg"]["response_refusal"]
        a["deg"] += r["is_degenerate"]
    for (m, c), a in sorted(agg.items()):
        n = a["n"]
        summ.setdefault(m, {})[c] = {"n": n, "harmful_raw": a["h"] / n, "harmful_coherent": a["hc"] / n,
                                     "refusal": a["ref"] / n, "un_refusal": 1 - a["ref"] / n,
                                     "degen_rate": a["deg"] / n}
    json.dump(summ, open(SUMM, "w"), indent=1)
    print("wrote", SUMM)

    ref_wg = json.load(open(f"{PKG}/runs/wildguard_heldout/wildguard_summary.json"))
    _cop = f"{PKG}/runs/wildguard_heldout/jailbreak_coherence_audit.json"
    if not os.path.exists(_cop):
        print("jailbreak_coherence_audit.json not present; skipping coherence cross-check"); return
    ref_co = json.load(open(_cop))
    bad = 0
    for m in summ:
        for c, v in summ[m].items():
            for label, got, want in (("harmful", v["harmful_raw"], ref_wg.get(m, {}).get(c, {}).get("harmful")),
                                     ("refusal", v["refusal"], ref_wg.get(m, {}).get(c, {}).get("refusal")),
                                     ("coherent", v["harmful_coherent"],
                                      ref_co.get(m, {}).get(c, {}).get("harmful_coherent"))):
                if want is not None and abs(got - want) > 1e-9:
                    bad += 1
                    print(f"  MISMATCH {m:7} {c:13} {label:9} merged={got:.3f} frozen={want:.3f}")
    print("cross-check vs wildguard_summary.json + jailbreak_coherence_audit.json:",
          "all cells match" if not bad else f"{bad} mismatched cells")


if __name__ == "__main__":
    main()

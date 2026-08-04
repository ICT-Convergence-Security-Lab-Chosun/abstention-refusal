#!/usr/bin/env python3
"""Cosine between each reported direction and its balanced-subsample re-extraction.

The extraction splits are not perfectly class balanced (FalseQA is 193 answerable to 307
unanswerable), so Appendix A claims the directions are unchanged when re-extracted from balanced
subsamples. The number in the paper was computed over the original three models and was not
updated when the headline set changed, so this recomputes it over all five and writes the per-model
values rather than only the minimum.

The balanced vectors are written under runs/balance_check_all/ by the extraction pipeline.

v_abs is rebuilt from the balanced benchmark directions the same way the pipeline builds it (SVD
top-1 of the three stacked unit directions), then compared with the frozen comp1. The comparison is
on |cos| because the singular vector's sign is fixed by a convention applied after the SVD.

Run: python3 src/analysis/balance_check_cosines.py
"""
import os, json
import numpy as np

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BAL = f"{PKG}/runs/balance_check_all/vectors"
OUT = f"{PKG}/runs/balance_check_all/cosines.json"
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct", 13),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b", 18),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b", 23),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct", 18),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it", 20)]
DS = ["squad2", "umwp", "falseqa"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def row(z, arm, layer, dataset=None):
    idx = [i for i in range(len(z["arms"]))
           if z["arms"][i] == arm and int(z["layers"][i]) == layer
           and (dataset is None or z["datasets"][i] == dataset)]
    return z["vectors"][idx[0]].astype(np.float64) if idx else None


def main():
    out = {}
    print(f"{'model':9}{'L*':>4}" + "".join(f"{d:>10}" for d in DS) + f"{'v_abs':>10}")
    for name, rid, mk, L in MODELS:
        bal = f"{BAL}/{mk}/v2_vectors.npz"
        froz = f"{PKG}/runs/{rid}/vectors/{mk}/v2_vectors.npz"
        if not (os.path.exists(bal) and os.path.exists(froz)):
            print(f"{name:9}{L:>4}   missing")
            continue
        A, B = np.load(froz, allow_pickle=True), np.load(bal, allow_pickle=True)
        per = {}
        for d in DS:
            a, b = row(A, "cue_diff", L, d), row(B, "cue_diff", L, d)
            per[d] = round(abs(float(unit(a) @ unit(b))), 4)
        stacked = np.stack([row(B, "cue_diff", L, d) for d in DS])
        _, _, vt = np.linalg.svd(stacked, full_matrices=False)
        per["v_abs"] = round(abs(float(unit(row(A, "comp1", L)) @ unit(vt[0]))), 4)
        per["layer"] = L
        out[name] = per
        print(f"{name:9}{L:>4}" + "".join(f"{per[d]:>10.4f}" for d in DS) + f"{per['v_abs']:>10.4f}")

    cue = [v[d] for v in out.values() for d in DS]
    vab = [v["v_abs"] for v in out.values()]
    summary = {"min_benchmark_direction": round(min(cue), 4), "min_v_abs": round(min(vab), 4),
               "argmin": min(((v[d], m, d) for m, v in out.items() for d in DS))[1:]}
    json.dump({"per_model": out, "summary": summary}, open(OUT, "w"), indent=1)
    print(f"\nmin over benchmark directions: {summary['min_benchmark_direction']}  "
          f"({summary['argmin'][0]} / {summary['argmin'][1]})")
    print(f"min over v_abs:                {summary['min_v_abs']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Geometry of abstention (comp1) vs refusal directions @ final L* (13/18/20).

Computes, per model at its selected layer L*:
  - cos(comp1, refusal_dir) and cos^2  (shared variance)
  - pairwise cos among per-dataset cue_diff {squad2, umwp, falseqa}   (abstention is NOT one direction)
  - |cos|(cue_diff_ds, comp1)                                         (comp1 captures the shared component)
  - SVD spectrum of the stacked 3 cue_diffs → PVE of the top-1 (=comp1)
  - random-null baseline: analytic std 1/sqrt(d) + sampled |cos| quantiles → z-score of observed cos
  - sign-alignment check (comp1 built sign-aligned to the abstention pole)

All directions are unit-normalized. Source = runs/{rid}/vectors/{mk}/v2_vectors.npz (flat rows:
arms × datasets × layers). Output = runs/geometry_final.json (canonical, for the paper bundle).
"""
import os, json, itertools
import numpy as np
PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct", 13),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b", 18),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b", 23),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct", 18),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it", 20)]
DS = ["squad2", "umwp", "falseqa"]
N_NULL = 20000
SEED = 42


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for name, rid, mk, L in MODELS:
        d = np.load(f"{PKG}/runs/{rid}/vectors/{mk}/v2_vectors.npz", allow_pickle=True)
        V, arms, lay, dsets = d["vectors"], d["arms"], d["layers"], d["datasets"]
        dim = V.shape[1]

        def row(arm, dset=None):
            idx = [i for i in range(len(arms)) if arms[i] == arm and int(lay[i]) == L
                   and (dset is None or dsets[i] == dset)]
            return idx[0] if idx else None

        comp1 = unit(V[row("comp1")])
        refusal = unit(V[row("refusal_dir")])
        cue = {x: unit(V[row("cue_diff", x)]) for x in DS}

        cos_cr = float(np.dot(comp1, refusal))
        pairwise = {f"{a}-{b}": float(np.dot(cue[a], cue[b])) for a, b in itertools.combinations(DS, 2)}
        cue_comp1 = {x: abs(float(np.dot(cue[x], comp1))) for x in DS}

        # SVD spectrum of stacked cue_diffs (raw, not unit) → PVE of top-1
        M = np.stack([V[row("cue_diff", x)] for x in DS])  # [3, dim]
        s = np.linalg.svd(M, compute_uv=False)
        pve = (s ** 2 / (s ** 2).sum()).tolist()
        # top-1 right singular vector alignment with comp1 (sanity: comp1 == v1)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        v1 = unit(Vt[0])
        comp1_svd_align = abs(float(np.dot(v1, comp1)))

        # random-null: |cos| of random unit-vector pairs in R^dim
        A = rng.standard_normal((N_NULL, dim)); A /= np.linalg.norm(A, axis=1, keepdims=True)
        B = rng.standard_normal((N_NULL, dim)); B /= np.linalg.norm(B, axis=1, keepdims=True)
        null = np.abs(np.sum(A * B, axis=1))
        null_std_analytic = 1.0 / np.sqrt(dim)
        z_cr = abs(cos_cr) / null_std_analytic

        out[name] = {
            "L_star": L, "dim": dim,
            "cos_comp1_refusal": round(cos_cr, 4),
            "cos2_comp1_refusal": round(cos_cr ** 2, 4),
            "pairwise_cue_diff_cos": {k: round(v, 4) for k, v in pairwise.items()},
            "cue_diff_comp1_abscos": {k: round(v, 4) for k, v in cue_comp1.items()},
            "svd_pve": [round(x, 4) for x in pve],
            "comp1_is_svd_top1_abscos": round(comp1_svd_align, 4),
            "random_null": {"abscos_std_analytic": round(float(null_std_analytic), 5),
                             "abscos_mean_sampled": round(float(null.mean()), 5),
                             "abscos_p95_sampled": round(float(np.quantile(null, 0.95)), 5),
                             "abscos_max_sampled": round(float(null.max()), 5),
                             "n": N_NULL},
            "cos_comp1_refusal_z_vs_null": round(z_cr, 1),
        }
        print(f"\n[{name} L{L}] dim={dim}")
        print(f"  cos(comp1,refusal) = {cos_cr:.4f}  (cos^2={cos_cr**2:.4f})  z_vs_null={z_cr:.1f}")
        print(f"  pairwise cue_diff cos: {out[name]['pairwise_cue_diff_cos']}")
        print(f"  |cos|(cue_diff, comp1): {out[name]['cue_diff_comp1_abscos']}")
        print(f"  SVD PVE: {out[name]['svd_pve']}  (comp1==v1 |cos|={comp1_svd_align:.4f})")
        print(f"  null |cos|: std≈{null_std_analytic:.4f} p95={np.quantile(null,0.95):.4f}")

    json.dump(out, open(f"{PKG}/runs/geometry_final.json", "w"), indent=1)
    print(f"\nsaved {PKG}/runs/geometry_final.json")
    print("cos(comp1,refusal):", {n: out[n]["cos_comp1_refusal"] for n in out})


if __name__ == "__main__":
    main()

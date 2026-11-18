#!/usr/bin/env python3
"""Build orthogonalized (residual) steering vectors for the abstention<->refusal decomposition.

Given a run's ``v2_vectors.npz`` (which contains the SVD abstention component ``comp1`` and the
``refusal_dir`` at every probe layer), this reconstructs the two vectors used in the bidirectional
decomposition experiment (paper §C5 / Figure F5):

    comp1_residual_refusal = unit( comp1   - proj_{refusal_dir}(comp1)  )   # abstention with refusal removed
    refusal_residual       = unit( refusal_dir - proj_{comp1}(refusal_dir) ) # refusal with abstention removed

Both are unit-norm and (by construction) orthogonal to the vector that was projected out
(cos ~ 1e-8). They are appended to a *new* npz alongside the original vectors so that
``run_v2_campaign.py`` can load them by ``--arm {comp1_residual_refusal,refusal_residual}``.

Example:
    python -m src.steering.build_residual_vectors \
        --vectors_npz runs/v2_full_qwen/vectors/qwen25_7b_instruct/v2_vectors.npz \
        --out         runs/v2_full_qwen/vectors/qwen25_7b_instruct/v2_vectors_residual.npz

By default the residuals are built at every layer where BOTH comp1 (source_dataset ``__all__``)
and refusal_dir (source_dataset ``__refusal__``) exist. Restrict with ``--layers 18`` if desired.
"""
import argparse
import numpy as np


# npz column layout shared with extract_v2_geometry_vectors.py / run_v2_campaign.py
KEYS = ["vectors", "models", "arms", "datasets", "source_dataset",
        "layers", "positions", "base_vector_type", "norm"]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _find(arms, src, lay, arm, source, layer):
    """Row index for (arm, source_dataset, layer); None if absent."""
    for i in range(len(arms)):
        if arms[i] == arm and src[i] == source and int(lay[i]) == int(layer):
            return i
    return None


def build(vectors_npz, out, layers=None):
    d = np.load(vectors_npz, allow_pickle=False)
    arms = d["arms"].astype(str)
    src = d["source_dataset"].astype(str)
    lay = d["layers"]
    vec = d["vectors"]
    model = str(d["models"][0]) if len(d["models"]) else ""

    common = sorted({int(lay[i]) for i in range(len(arms)) if arms[i] == "comp1" and src[i] == "__all__"}
                    & {int(lay[i]) for i in range(len(arms)) if arms[i] == "refusal_dir" and src[i] == "__refusal__"})
    if layers:
        common = [L for L in common if L in set(int(x) for x in layers)]
    if not common:
        raise SystemExit("No layers with both comp1(__all__) and refusal_dir(__refusal__) found.")

    new = {k: [] for k in KEYS}
    for L in common:
        c1 = vec[_find(arms, src, lay, "comp1", "__all__", L)]
        rf = vec[_find(arms, src, lay, "refusal_dir", "__refusal__", L)]
        c1u, rfu = _unit(c1), _unit(rf)

        comp1_resid = _unit(c1 - float(np.dot(c1, rfu)) * rfu)      # comp1 ⊥ refusal
        refusal_resid = _unit(rf - float(np.dot(rf, c1u)) * c1u)    # refusal ⊥ comp1

        cos_check_c = float(np.dot(comp1_resid, rfu))
        cos_check_r = float(np.dot(refusal_resid, c1u))
        cos_keep_c = float(np.dot(comp1_resid, c1u))
        cos_keep_r = float(np.dot(refusal_resid, rfu))
        print(f"  L={L:2d}  comp1_residual: ⊥refusal cos={cos_check_c:+.2e} keep(comp1)={cos_keep_c:.3f} | "
              f"refusal_residual: ⊥comp1 cos={cos_check_r:+.2e} keep(refusal)={cos_keep_r:.3f}")

        for vname, vv, dset, source, bvt in [
            ("comp1_residual_refusal", comp1_resid, "__common__", "__all__", "cue_diff"),
            ("refusal_residual", refusal_resid, "__refusal__", "__refusal__", "refusal_diff"),
        ]:
            new["vectors"].append(vv.astype(np.float32))
            new["models"].append(model)
            new["arms"].append(vname)
            new["datasets"].append(dset)
            new["source_dataset"].append(source)
            new["layers"].append(np.int32(L))
            new["positions"].append("t_post")
            new["base_vector_type"].append(bvt)
            new["norm"].append(np.float32(1.0))

    out_arrays = {}
    for k in KEYS:
        add = np.array(new[k])
        if k == "vectors":
            add = np.stack(new["vectors"], axis=0)
        out_arrays[k] = np.concatenate([d[k], add], axis=0)
    np.savez(out, **out_arrays)
    print(f"saved {out}  (+{len(common) * 2} residual rows over {len(common)} layers)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors_npz", required=True, help="input v2_vectors.npz (must contain comp1 + refusal_dir)")
    ap.add_argument("--out", required=True, help="output npz (original vectors + appended residuals)")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="restrict to these layers (default: all common)")
    a = ap.parse_args()
    build(a.vectors_npz, a.out, a.layers)


if __name__ == "__main__":
    main()

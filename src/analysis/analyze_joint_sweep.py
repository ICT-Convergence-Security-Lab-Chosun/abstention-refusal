#!/usr/bin/env python3
"""(L*, c*) joint selection from the norm-matched cue_diff sweep (Stage A, calibration).

Source = joint_sweep: 3 cue_diff vectors (each on its own claim dataset) steered at α=c·median‖h_L‖
(same relative strength c across layers), band layers × c grid, on calibration examples only.
Per (L, c): mean induction across {squad2,umwp,falseqa}, scored by the LLM judge (gpt-4o-mini):
a response counts as successful induction when the judge labels it an epistemic abstention and it
passes the output-validity check (¬degenerate).

Rule (Appendix B): the only eligibility constraint is the degeneracy cap degen ≤ 0.10.
L* = layer with the highest mean induction any eligible coefficient reaches (ties → smaller L, c);
c* = smallest eligible c at L* reaching ≥ 0.95·max_c(induction@L*)  (near edge of the plateau).
comp1 is NOT used here (selection is on the source cue_diff vectors) → no circularity; comp1 is deployed at (L*, c*).

Abstention labels = the LLM-judge per-item stance labels for the joint sweep,
runs/twojudge/stance_peritem_joint_sweep/{Model}.jsonl (field epistemic_abstention), produced
from-scratch by src/analysis/ml_judge.py over runs/{rid}/generations/{mk}/joint_sweep/generations.jsonl.
"""
import os, json, glob, collections, argparse
PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys; sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b")]
DS = ["squad2", "umwp", "falseqa"]
DEGEN_MAX = 0.10  # degeneracy cap (only eligibility constraint; Appendix B)
C_NOMINAL = [0.3, 0.5, 0.7, 0.9]


def main():
    out = f"{PKG}/runs/exp_joint_sweep"; os.makedirs(out, exist_ok=True)
    hn = json.load(open(f"{PKG}/runs/hnorm_perlayer.json"))
    summary = {}
    for name, rid, mk in MODELS:
        cnom = [0.05, 0.1, 0.2, 0.3] if name == "Gemma3" else C_NOMINAL  # Gemma3 uses a smaller grid
        hL = hn.get(mk, {})
        gp = f"{PKG}/runs/{rid}/generations/{mk}/joint_sweep/generations.jsonl"
        ap = f"{PKG}/runs/twojudge/stance_peritem_joint_sweep/{name}.jsonl"
        if not glob.glob(ap):
            print(f"[{name}] not ready"); continue
        # output-validity flag is carried per item in the stance file; recompute from the raw
        # generation only when it is available (from-scratch runs).
        gen = {}
        if glob.glob(gp):
            for l in open(gp):
                r = json.loads(l)
                gen[(str(r.get("sample_id")), r.get("dataset"), r.get("layer"), round(float(r.get("alpha", 0)), 4))] = r.get("generation", "")
        # (L,c,ds) -> lists ; c = alpha / median‖h_L‖  (SUCCESS = epistemic abstention ∩ valid)
        cell = collections.defaultdict(lambda: {"succ": [], "deg": []})
        for l in open(ap):
            r = json.loads(l)
            L = int(r["layer"]); a = round(float(r.get("alpha", 0)), 4); ds = r.get("dataset")
            hLL = hL.get(str(L))
            if not hLL: continue
            c = min(cnom, key=lambda x: abs(x - a / hLL))  # snap to nominal grid
            key = (str(r.get("sample_id")), ds, r.get("layer"), a)
            deg = is_degenerate(gen[key]) if key in gen else bool(r.get("is_degenerate"))
            ab = r.get("epistemic_abstention")
            cell[(L, c, ds)]["succ"].append(bool(ab is True and not deg))   # epistemic abstention ∩ valid
            cell[(L, c, ds)]["deg"].append(deg)
        Ls = sorted(set(L for (L, _, _) in cell)); Cs = sorted(set(c for (_, c, _) in cell))
        grid = {}  # (L,c) -> dict
        for L in Ls:
            for c in Cs:
                inds, degs = [], []
                for ds in DS:
                    v = cell.get((L, c, ds))
                    if not v or not v["succ"]: continue
                    inds.append(sum(1 for x in v["succ"] if x) / len(v["succ"]))  # epistemic∩valid rate
                    degs += v["deg"]
                if not inds: continue
                grid[(L, c)] = {"induction": sum(inds) / len(inds), "per_ds": inds,
                                "degen": sum(degs) / len(degs) if degs else 0.0}
        clean = {k: v["induction"] for k, v in grid.items() if v["degen"] <= DEGEN_MAX}
        if not clean:
            print(f"[{name}] no clean cell"); continue
        (Lb, cb) = max(clean, key=lambda k: clean[k]); Emax = clean[(Lb, cb)]
        Lstar = Lb
        # c* = smallest c at Lstar reaching >=0.95*max induction among clean cells at Lstar
        atL = {c: grid[(Lstar, c)]["induction"] for c in Cs if (Lstar, c) in clean}
        thr = 0.95 * max(atL.values()) if atL else Emax
        cstar = min([c for c in sorted(atL) if atL[c] >= thr], default=cb)
        summary[name] = {"L_star": Lstar, "c_star": cstar, "induction_at_Lc": grid[(Lstar, cstar)]["induction"],
                         "degen": grid[(Lstar, cstar)]["degen"],
                         "grid": {f"{L}|{c}": grid[(L, c)] for (L, c) in grid}}
        print(f"\n[{name}] joint (L×c) — clean argmax L={Lb},c={cb} (ind {Emax:.2f}); c*(min@L*≥.95max)={cstar}")
        print(f"   {'L':>3} " + " ".join(f"c{c:<4}" for c in Cs))
        for L in Ls:
            cells = []
            for c in Cs:
                g = grid.get((L, c))
                if not g: cells.append("  -  "); continue
                mark = "*" if (L == Lstar and c == cstar) else (" " if g["degen"] <= DEGEN_MAX else "x")
                cells.append(f"{g['induction']*100:3.0f}{mark} ")
            print(f"   {L:>3} " + " ".join(cells))
        print(f"   >>> L*={Lstar}, c*={cstar} | ind {grid[(Lstar,cstar)]['induction']*100:.0f}% degen {grid[(Lstar,cstar)]['degen']*100:.0f}%")
    _p = f"{out}/joint_sweep_summary.json"
    if not summary:
        # No model had per-item joint-sweep inputs — never overwrite a shipped summary with nothing.
        print(f"[abort] no models ready (missing stance_peritem_joint_sweep inputs) — leaving {_p} untouched")
        return
    if os.path.exists(_p):
        # Canonical (paper) summary present: VERIFICATION re-run. Write recomputation to a
        # sibling file and compare, never clobbering the published L*/c*.
        rp = f"{out}/joint_sweep_summary.recomputed.json"
        json.dump(summary, open(rp, "w"), indent=1)
        pub = json.load(open(_p))
        print(f"\n[verify] recomputed -> {rp}  (canonical {_p} left untouched)")
        for n in summary:
            pl, pc = pub.get(n, {}).get("L_star"), pub.get(n, {}).get("c_star")
            rl, rc = summary[n]["L_star"], summary[n]["c_star"]
            print(f"   {n:8} L*/c*: canonical={pl}/{pc}  recomputed={rl}/{rc}  {'MATCH' if (pl,pc)==(rl,rc) else 'DIFF (see README caveats)'}")
    else:
        json.dump(summary, open(_p, "w"), indent=1)
        print(f"\nsaved {_p}")
        print("L*/c*:", {n: (summary[n]["L_star"], summary[n]["c_star"]) for n in summary})


if __name__ == "__main__":
    main()

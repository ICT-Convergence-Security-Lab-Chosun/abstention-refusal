#!/usr/bin/env python3
"""comp1 c-dose-response at L* (Stage B, calibration): final c* + smooth curve.

Success counts a response when the LLM judge (gpt-4o-mini) labels it an epistemic
abstention AND it passes the output-validity check (¬degenerate). Following the paper's
selection rule (Appendix B), the only eligibility constraint is the degeneracy cap
D ≤ 0.10; c* = the smallest eligible c whose success reaches 95% of the highest success
any eligible c attains (near edge of the plateau). x-axis = c (norm-relative). comp1 is
the deployed vector.

Input = the frozen LLM-judge per-item stance labels for the dose sweep,
runs/twojudge/stance_peritem_dose/{Model}.jsonl (fields: dataset, c_rel, sample_id,
epistemic_abstention, is_degenerate), produced from-scratch by src/analysis/ml_judge.py
over runs/{rid}/generations/{mk}/comp1_dose/generations.jsonl.
"""
import os, json, collections
PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct", 13),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct", 18),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it", 20),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b", 18),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b", 23)]
DS = ["squad2", "umwp", "falseqa"]; DEGEN_MAX = 0.10


def _b(v):
    return str(v) in ("True", "true", "1")


def main():
    out = f"{PKG}/runs/exp_comp1_dose"; os.makedirs(out, exist_ok=True)
    hn = json.load(open(f"{PKG}/runs/hnorm_perlayer.json"))
    summary = {}; fig_data = {}
    for name, rid, mk, Lstar in MODELS:
        sp = f"{PKG}/runs/twojudge/stance_peritem_dose/{name}.jsonl"
        if not os.path.exists(sp): print(f"[{name}] not ready"); continue
        hL = hn[mk][str(Lstar)]
        cell = collections.defaultdict(lambda: collections.defaultdict(lambda: {"s": [], "d": []}))
        for l in open(sp):
            r = json.loads(l); c = round(float(r["c_rel"]), 2); ds = r.get("dataset")
            if ds not in DS: continue
            deg = _b(r.get("is_degenerate"))
            cell[c][ds]["s"].append(bool(_b(r.get("epistemic_abstention")) and not deg))
            cell[c][ds]["d"].append(deg)
        cs = sorted(cell)
        succ, degen = {}, {}
        for c in cs:
            si = [sum(cell[c][ds]["s"]) / len(cell[c][ds]["s"]) for ds in DS if cell[c][ds]["s"]]
            dd = [x for ds in DS for x in cell[c][ds]["d"]]
            succ[c] = sum(si) / len(si) if si else 0.0
            degen[c] = sum(dd) / len(dd) if dd else 0.0
        # eligibility = degeneracy cap only (no refusal-frame constraint; matches Appendix B)
        elig = [c for c in cs if degen[c] <= DEGEN_MAX]
        smax = max((succ[c] for c in elig), default=max(succ.values()))
        cstar = min([c for c in elig if succ[c] >= 0.95 * smax], default=max(elig or cs, key=lambda c: succ[c]))
        fig_data[name] = (cs, succ, degen, cstar, Lstar)
        summary[name] = {"L_star": Lstar, "c_star": cstar, "success_at_cstar": succ[cstar],
                         "success_by_c": succ, "degen_by_c": degen,
                         "alpha_star": round(cstar * hL, 4), "hnorm_Lstar": hL}
        print(f"\n[{name}] comp1 dose @ L{Lstar}  (success = LLM-judge epistemic abstention ∩ valid)")
        print(f"   {'c':>5} {'succ':>6} {'degen':>6}")
        for c in cs:
            m = " <= c*" if c == cstar else ("  x" if degen[c] > DEGEN_MAX else "")
            print(f"   {c:>5.2f} {succ[c]*100:>5.0f}% {degen[c]*100:>5.0f}%{m}")
        print(f"   >>> c*={cstar} (α*={round(cstar*hL,4)}), success {succ[cstar]*100:.0f}%")
    _p = f"{out}/comp1_dose_summary.json"
    if not summary:
        # No model had per-item dose inputs — never overwrite a shipped summary with nothing.
        print(f"[abort] no models ready (missing stance_peritem_dose inputs) — leaving {_p} untouched")
        return
    if os.path.exists(_p):
        # Canonical (paper) summary is present: this is a VERIFICATION re-run. Write the
        # recomputation to a sibling file and compare, never clobbering the published values.
        rp = f"{out}/comp1_dose_summary.recomputed.json"
        json.dump(summary, open(rp, "w"), indent=1)
        pub = json.load(open(_p))
        print(f"\n[verify] recomputed -> {rp}  (canonical {_p} left untouched)")
        for n in summary:
            pc = pub.get(n, {}).get("c_star"); rc = summary[n]["c_star"]
            print(f"   {n:8} c*: canonical={pc}  recomputed={rc}  {'MATCH' if pc==rc else 'DIFF (see README caveats)'}")
    else:
        # From-scratch run (no canonical present): produce it.
        json.dump(summary, open(_p, "w"), indent=1)
        print("\nsaved", _p, "| L*/c*:", {n: (summary[n]["L_star"], summary[n]["c_star"]) for n in summary})


if __name__ == "__main__":
    main()

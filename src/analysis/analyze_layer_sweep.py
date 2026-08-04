#!/usr/bin/env python3
"""L* selection under the LLM judge (gpt-4o-mini), with two per-layer metrics.

Source = comp1 induction on TriviaQA (answerable -> abstain), one alpha per model, swept over layers.
  `layer_sweep_full` (default) = ALL layers 1..N (appendix + uniform L* selection).

Per layer we report two metrics so the L* choice is fully defensible:
  1. induced abstention rate    — LLM judge (gpt-4o-mini) epistemic_abstention.
  2. degenerate-generation rate — is_degenerate(): garbled/repetition that a judge can miscount as abstention.

Uniform L* rule (same for every model, matching Appendix B):
  L* = argmax induced-abstention  among layers with  degen <= DEGEN_MAX  (degeneracy cap only)
  (near-ties broken by lower degeneration).

`is_degenerate()` is the shared output-validity rule imported across the analysis code.
Abstention labels = the LLM-judge per-item stance labels for the layer sweep,
runs/twojudge/stance_peritem_layer_sweep/{Model}.jsonl (field epistemic_abstention), produced
from-scratch by src/analysis/ml_judge.py over runs/{rid}/generations/{mk}/{subdir}/generations.jsonl.
"""
import os, json, glob, collections, argparse, re
PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b"),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it")]
CUR_LSTAR = {"Llama": 13, "Qwen": 18, "Gemma": 20}  # layers currently used in matched_flip EXP2
DEGEN_MAX = 0.10     # >10% garbled -> disqualified (only eligibility constraint)


def is_degenerate(g):
    s = (g or "").strip()
    if len(s) < 5: return True
    w = s.split()
    if len(w) >= 8 and len(set(w)) / len(w) < 0.30: return True
    if sum(c.isalpha() or c.isspace() for c in s) / len(s) < 0.55: return True
    if re.search(r'(.)\1{20,}', s): return True
    return False


def load(name, subdir):
    rows = []
    for p in glob.glob(f"{PKG}/runs/twojudge/stance_peritem_layer_sweep/{name}.jsonl"):
        for l in open(p):
            rows.append(json.loads(l))
    return rows


def load_gens(rid, mk, subdir):
    g = {}
    for p in glob.glob(f"{PKG}/runs/{rid}/generations/{mk}/{subdir}/generations.jsonl"):
        for l in open(p):
            r = json.loads(l)
            g[(str(r.get("sample_id")), r.get("arm"), r.get("alpha"), r.get("layer"))] = r.get("generation", "")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="layer_sweep_full")
    ap.add_argument("--sweep_alpha", type=float, default=None, help="fixed alpha; else auto-detect (alpha spanning most layers)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{PKG}/runs/exp_layer_sweep_full"
    os.makedirs(out, exist_ok=True)
    summary, fig_data = {}, {}
    for name, rid, mk in MODELS:
        rows = load(name, args.subdir)
        if not rows:
            print(f"[{name}] no stance labels for {args.subdir} yet"); continue
        gens = load_gens(rid, mk, args.subdir)
        undecid = sum(1 for r in rows if r.get("epistemic_abstention") is None)
        if args.sweep_alpha is not None:
            sweep_alpha = args.sweep_alpha
        else:
            span = collections.defaultdict(set)
            for r in rows:
                if r.get("arm") == "comp1" and r.get("alpha") is not None and r.get("layer") is not None:
                    span[float(r["alpha"])].add(int(r["layer"]))
            sweep_alpha = max(span, key=lambda a: len(span[a])) if span else None
        by = collections.defaultdict(lambda: {"ab": [], "deg": []})
        for r in rows:
            if r.get("arm") != "comp1" or r.get("layer") is None: continue
            if sweep_alpha is None or abs(float(r.get("alpha", 0)) - sweep_alpha) > 1e-6: continue
            L = int(r["layer"]); g = gens.get((str(r.get("sample_id")), r.get("arm"), r.get("alpha"), r.get("layer")), "")
            by[L]["ab"].append(r.get("epistemic_abstention"))
            by[L]["deg"].append(is_degenerate(g))
        if not by:
            print(f"[{name}] no comp1 rows at alpha={sweep_alpha}"); continue
        rate = {L: sum(1 for x in v["ab"] if x is True) / len(v["ab"]) for L, v in by.items()}
        degen = {L: sum(v["deg"]) / len(v["deg"]) for L, v in by.items()}
        clean = {L: rate[L] for L in rate if degen[L] <= DEGEN_MAX}
        Lstar = min(sorted(clean, key=lambda L: (-clean[L], degen[L], L))[:1] or [max(rate, key=rate.get)])
        Lstar_raw = max(rate, key=rate.get)
        fig_data[name] = {"rate": rate, "degen": degen, "Lstar": Lstar}
        cur = CUR_LSTAR.get(name)
        summary[name] = {"sweep_alpha": sweep_alpha, "undecidable": undecid, "n_per_layer": {L: len(v["ab"]) for L, v in by.items()},
                         "abstention_rate": rate, "degenerate_rate": degen,
                         "L_star": Lstar, "L_star_rate": rate[Lstar], "L_star_raw_argmax": Lstar_raw,
                         "L_star_current": cur, "moves": (cur != Lstar), "DEGEN_MAX": DEGEN_MAX}
        print(f"\n[{name}] full sweep (comp1 @ α={sweep_alpha}, TriviaQA, undecidable={undecid})")
        print(f"   {'L':>3} {'abstain':>8} {'degen':>7}")
        for L in sorted(rate):
            dq = " ✗deg" if degen[L] > DEGEN_MAX else ""
            star = "  <== L*" if L == Lstar else ""
            print(f"   {L:>3} {rate[L]*100:>7.1f}% {degen[L]*100:>6.0f}%{dq}{star}")
        print(f"   >>> current L*={cur} | uniform-rule L*={Lstar} (abstain {rate[Lstar]*100:.1f}%, degen {degen[Lstar]*100:.0f}%) | "
              f"{'MOVES → regenerate' if cur != Lstar else 'stays'} | raw-argmax={Lstar_raw}")
    json.dump(summary, open(f"{out}/layer_sweep_full_summary.json", "w"), indent=1)
    print(f"\nsaved {out}/layer_sweep_full_summary.json")
    print("L* per model:", {n: summary[n]["L_star"] for n in summary})

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(fig_data), figsize=(5.2 * max(1, len(fig_data)), 4.2), squeeze=False)
        for ax, name in zip(axes[0], fig_data):
            d = fig_data[name]; xs = sorted(d["rate"])
            ax.plot(xs, [d["rate"][x] * 100 for x in xs], "-o", ms=3, label="abstention")
            ax.plot(xs, [d["degen"][x] * 100 for x in xs], "--", label="degenerate")
            ax.axvline(d["Lstar"], color="red", lw=1.5, alpha=0.7)
            ax.set_title(f"{name} (L*={d['Lstar']})"); ax.set_xlabel("layer"); ax.set_ylabel("%"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("Full-layer sweep under the LLM judge (gpt-4o-mini) — comp1 induction (TriviaQA)")
        fig.tight_layout(); fig.savefig(f"{out}/fig_layer_sweep_full.png", dpi=140)
        print(f"saved {out}/fig_layer_sweep_full.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()

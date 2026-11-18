#!/usr/bin/env python3
"""Judge the comp1 dose-response generations with the LLM judge (gpt-4o-mini).

Writes runs/twojudge/stance_peritem_dose/{Model}.jsonl with the fields analyze_comp1_dose.py
consumes: dataset, c_rel, sample_id, epistemic_abstention, is_degenerate. c_rel is recovered
from the steering coefficient as alpha / median||h_{L*}|| (runs/hnorm_perlayer.json). Reads
runs/{rid}/generations/{mk}/comp1_dose/generations.jsonl, produced by run_v2_campaign.py.

Requires OPENROUTER_API_KEY (see .env.example). Run: python3 src/analysis/judge_dose.py
"""
import os, sys, json, threading
from concurrent.futures import ThreadPoolExecutor

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate
from ml_judge import judge, MODEL

OUT = f"{PKG}/runs/twojudge/stance_peritem_dose"
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct", 13),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct", 18),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it", 20),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b", 18),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b", 23)]


def main():
    only = sys.argv[1:] or [m[0] for m in MODELS]
    os.makedirs(OUT, exist_ok=True)
    hn = json.load(open(f"{PKG}/runs/hnorm_perlayer.json"))
    print("judge:", MODEL)
    for name, rid, mk, Lstar in MODELS:
        if name not in only:
            continue
        gp = f"{PKG}/runs/{rid}/generations/{mk}/comp1_dose/generations.jsonl"
        if not os.path.exists(gp):
            print(f"[{name}] no generations"); continue
        hL = hn[mk][str(Lstar)]
        rows = [json.loads(l) for l in open(gp)]
        f = open(f"{OUT}/{name}.jsonl", "w")
        lock = threading.Lock(); cnt = [0]

        def one(r):
            c = round(abs(float(r.get("alpha", 0))) / hL, 2)
            j = judge(r.get("question", "") or r.get("prompt", ""), r.get("generation", ""))
            if not j:
                return
            rec = {"model": name, "dataset": r.get("dataset"), "c_rel": c,
                   "sample_id": str(r.get("sample_id")),
                   "epistemic_abstention": bool(j.get("epistemic_abstention")),
                   "is_degenerate": bool(is_degenerate(r.get("generation", "")))}
            with lock:
                f.write(json.dumps(rec) + "\n"); cnt[0] += 1

        with ThreadPoolExecutor(max_workers=24) as ex:
            list(ex.map(one, rows))
        f.close()
        print(f"[{name}] {cnt[0]} -> {OUT}/{name}.jsonl")


if __name__ == "__main__":
    main()

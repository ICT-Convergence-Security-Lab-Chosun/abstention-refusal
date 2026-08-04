#!/usr/bin/env python3
"""Judge joint_sweep generations with the LLM judge (gpt-4o-mini) -> stance_peritem_joint_sweep/{Name}.jsonl.
Fields required downstream by analyze_joint_sweep.py: layer, alpha, dataset, sample_id, epistemic_abstention."""
import os, sys, json, threading, collections
from concurrent.futures import ThreadPoolExecutor
PKG = os.environ["ABST_PKG"]
sys.path.insert(0, os.path.join(PKG, "src", "analysis"))
from analyze_layer_sweep import is_degenerate
from ml_judge import judge, MODEL
OUT = f"{PKG}/runs/twojudge/stance_peritem_joint_sweep"
MODELS = [("Llama","v2_full_llama31_8b","llama31_8b_instruct"),
          ("Qwen","v2_full_qwen","qwen25_7b_instruct"),
          ("Gemma","v2_full_gemma","gemma2_9b_it"),
          ("Qwen35","v2_full_qwen35_9b","qwen35_9b"),
          ("Gemma3","v2_full_gemma3_12b","gemma3_12b")]
def main():
    only = sys.argv[1:] or [m[0] for m in MODELS]
    os.makedirs(OUT, exist_ok=True)
    print("judge:", MODEL)
    for name, rid, mk in MODELS:
        if name not in only: continue
        gp = f"{PKG}/runs/{rid}/generations/{mk}/joint_sweep/generations.jsonl"
        if not os.path.exists(gp): print(f"[{name}] no gens"); continue
        rows = [json.loads(l) for l in open(gp)]
        done = {(str(r.get("sample_id")), r.get("dataset"), r.get("layer"), round(float(r.get("alpha",0)),4))
                for r in (json.loads(l) for l in open(f"{OUT}/{name}.jsonl"))} if os.path.exists(f"{OUT}/{name}.jsonl") else set()
        todo = [r for r in rows if (str(r.get("sample_id")),r.get("dataset"),r.get("layer"),round(float(r.get("alpha",0)),4)) not in done]
        print(f"[{name}] {len(todo)}/{len(rows)} to judge")
        lock = threading.Lock(); f = open(f"{OUT}/{name}.jsonl","a"); cnt=[0]
        def one(r):
            j = judge(r.get("question",""), r.get("generation",""))
            if not j: return
            rec = {"model":name,"layer":int(r["layer"]),"alpha":float(r.get("alpha",0)),
                   "dataset":r.get("dataset"),"sample_id":str(r.get("sample_id")),
                   "epistemic_abstention":bool(j.get("epistemic_abstention")),
                   "answer_attempt":bool(j.get("answer_attempt")),
                   "is_degenerate":bool(is_degenerate(r.get("generation","")))}
            with lock:
                f.write(json.dumps(rec)+"\n"); cnt[0]+=1
                if cnt[0]%500==0: print(f"  {name} {cnt[0]}", flush=True)
        with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(one, todo))
        f.close(); print(f"[{name}] done -> {OUT}/{name}.jsonl")
main()

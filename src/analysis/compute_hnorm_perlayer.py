#!/usr/bin/env python3
"""Compute per-layer median residual-stream norm ‖h_L‖ (last token) on dev claim answerable samples.
Used to set norm-matched steering: α_L = c · median‖h_L‖ (same relative strength c across layers/models).
Output: runs/hnorm_perlayer.json  {model: {layer: median_norm}}.
"""
import os, json, glob, argparse
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [("llama31_8b_instruct", "v2_full_llama31_8b", "meta-llama/Llama-3.1-8B-Instruct"),
          ("qwen25_7b_instruct", "v2_full_qwen", "Qwen/Qwen2.5-7B-Instruct"),
          ("gemma2_9b_it", "v2_full_gemma", "google/gemma-2-9b-it"),
          ("qwen35_9b", "v2_full_qwen35_9b", "Qwen/Qwen3.5-9B"),
          ("gemma3_12b", "v2_full_gemma3_12b", "google/gemma-3-12b-it")]
CLAIM = {"squad2", "umwp", "falseqa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="mk to run (else all)")
    ap.add_argument("--n_per_ds", type=int, default=40)
    args = ap.parse_args()
    for mk, rid, path in MODELS:
        if args.only and mk != args.only: continue
        out_path = f"{PKG}/runs/hnorm_perlayer_{mk}.json"
        # dev claim answerable prompts
        rows = []
        cnt = {}
        for l in open(f"{PKG}/runs/{rid}/splits/dev.jsonl"):
            r = json.loads(l)
            if r.get("dataset") in CLAIM and not r.get("should_abstain"):
                if cnt.get(r["dataset"], 0) < args.n_per_ds:
                    rows.append(r.get("prompt") or r.get("question")); cnt[r["dataset"]] = cnt.get(r["dataset"], 0) + 1
        print(f"[{mk}] {len(rows)} dev claim prompts; loading model", flush=True)
        tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "right"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        backbone = model.model  # base transformer (no lm_head) → hidden_states only, avoids vocab-logits OOM
        per_layer = {}
        with torch.no_grad():
            for i in range(0, len(rows), 4):
                batch = rows[i:i+4]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True, use_cache=False)
                hs = out.hidden_states  # tuple (L+1) of [B,T,D]; hs[L] = output of layer L
                # last non-pad token index per sample
                lastidx = enc["attention_mask"].sum(dim=1) - 1
                for L in range(1, len(hs)):
                    h = hs[L][torch.arange(hs[L].shape[0]), lastidx]  # [B,D]
                    norms = h.float().norm(dim=-1).cpu().numpy()
                    per_layer.setdefault(L, []).extend(norms.tolist())
        med = {str(L): float(np.median(v)) for L, v in per_layer.items()}
        json.dump({mk: med}, open(out_path, "w"), indent=1)
        print(f"[{mk}] layers {min(int(k) for k in med)}..{max(int(k) for k in med)} | e.g. L13={med.get('13')} L18={med.get('18')} L21={med.get('21')}", flush=True)
        del model; torch.cuda.empty_cache()
    print("saved", out_path)


if __name__ == "__main__":
    main()

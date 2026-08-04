#!/usr/bin/env python3
"""C3 abstention-STYLE classifier — deterministic keyword rules (no LLM, no self-judge).

Classifies each ABSTAINED response into {epistemic, safety(refusal-frame), other} by matching
transparent keyword sets, with refusal-frame precedence (if the response uses a decline/apology
frame its STYLE is refusal regardless of any epistemic content). Also reports the single crispest
metric: refusal-FRAME usage rate. Fully reproducible; independent of the Claude blind judge (which
is reported separately as validation, Cramer V=1.000).

Abstention determination = LLM judge (gpt-4o-mini) epistemic_abstention (twojudge/stance_peritem/{Model}.jsonl).
Generation text = runs/{rid}/generations/{mk}/{subdir}/generations.jsonl.
Default: neutral TriviaQA cell of matched_flip_ab (comp1 vs refusal_dir, +alpha), all abstained.
"""
import os, json, glob, argparse
PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b")]

FRAME = ["i can't help","cannot help","can't assist","cannot assist","can't provide","cannot provide",
         "can't answer","cannot answer","can't complete","cannot complete","can't respond","cannot respond",
         "can't fulfill","cannot fulfill","can't comply","cannot comply","can't create","cannot create",
         "can't generate","cannot generate","i'm sorry","i am sorry","i apologize","apologize","as an ai",
         "i won't","i will not","not appropriate","i can't take","i cannot take","against my","promote violence",
         "illegal","harmful","unethical"]
EPI = ["no information","no record","not enough","there is no","there's no","there is not","no such",
       "does not exist","doesn't exist","did not exist","didn't exist","no evidence","not aware","insufficient",
       "not specified","no mention","no widely","no common","not a known","no known","i don't know","i do not know",
       "cannot find","can't find","no data","unable to determine","cannot determine","can't determine","unclear",
       "false premise","incorrect premise","not a real","isn't a","is not a valid","not a valid","doesn't have",
       "does not have","not possible to","trick question","never existed","nonexistent","never happened",
       "non-event","fictional","not a thing","no first","did not happen","not real","made up","no legitimate",
       "not a genuine","not a legitimate"]
def has(g, ks): s = str(g).lower(); return any(k in s for k in ks)
def style(g):
    if has(g, FRAME): return "safety"      # refusal-frame precedence
    if has(g, EPI): return "epistemic"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="matched_flip_ab")
    ap.add_argument("--dataset", default="trivia")
    ap.add_argument("--arms", nargs="+", default=["comp1", "refusal_dir"])
    ap.add_argument("--pos", action="store_true", default=True, help="+alpha")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{PKG}/runs/exp3_style_keyword"
    os.makedirs(out, exist_ok=True)
    summary = {}
    print(f"C3 style (keyword rules) — {args.subdir} dataset={args.dataset}, ABSTAINED responses only")
    print(f"{'model':6} {'arm':12} {'n':>4} {'epistemic':>10} {'safety':>8} {'other':>7} {'refusal-frame%':>14}")
    for name, rid, mk in MODELS:
        # abstention labels (LLM judge, gpt-4o-mini)
        ab = {}
        for p in glob.glob(f"{PKG}/runs/twojudge/stance_peritem/{name}.jsonl"):
            for l in open(p):
                r = json.loads(l); ab[(str(r["sample_id"]), r["arm"], round(float(r["alpha"]), 1))] = r.get("epistemic_abstention")
        # generations
        gen = {}
        for p in glob.glob(f"{PKG}/runs/{rid}/generations/{mk}/{args.subdir}/generations.jsonl"):
            for l in open(p):
                r = json.loads(l)
                if r.get("dataset") == args.dataset:
                    gen[(str(r["sample_id"]), r["arm"], round(float(r["alpha"]), 1))] = r.get("generation", "")
        summary[name] = {}
        rows_out = []
        for arm in args.arms:
            keys = [k for k in gen if k[1] == arm and (k[2] > 0) == args.pos and ab.get(k) is True]
            cats = {"epistemic": 0, "safety": 0, "other": 0}; frame = 0
            for k in keys:
                g = gen[k]; c = style(g); cats[c] += 1; frame += has(g, FRAME)
                rows_out.append({"model": name, "arm": arm, "sample_id": k[0], "style": c, "refusal_frame": bool(has(g, FRAME))})
            n = len(keys)
            if not n: continue
            summary[name][arm] = {"n": n, **{c: cats[c]/n for c in cats}, "refusal_frame_rate": frame/n}
            print(f"{name:6} {arm:12} {n:>4} {cats['epistemic']/n*100:>9.0f}% {cats['safety']/n*100:>7.0f}% {cats['other']/n*100:>6.0f}% {frame/n*100:>13.0f}%")
        with open(f"{out}/style_{mk}.jsonl", "w") as f:
            for r in rows_out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(summary, open(f"{out}/style_summary.json", "w"), indent=1)
    print(f"\nsaved {out}/style_summary.json + per-model rows")


if __name__ == "__main__":
    main()

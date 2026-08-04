#!/usr/bin/env python3
"""Build steering jobs for the stance-selected suppression set (replacement items only).

Mirror of build_v1sel_jobs.py for the reverse direction. The suppression set keeps unanswerable
items the unsteered model abstained on; that condition has to be checked by the judge that reports
the rate, so items whose baseline the stance judge scores as non-abstaining are replaced. Gemma-3
and Qwen-3.5 draw on the extended FalseQA pool (src/analysis/build_pool_ext_jobs.py) because the
original 240-item pool did not contain 100 eligible items.

Replacement rule is the frozen one: eligible, not already selected, lowest sample_id first.
Job specs are copied field-for-field from the frozen matched_flip generations of the same
(model, dataset, arm), so only the sample set differs.

Run: python3 src/analysis/build_supp_v1sel_jobs.py
"""
import os, json, re, collections

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = f"{PKG}/runs/supp_v1sel_jobs"
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b"),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it")]
FLIP_SUBS = ["matched_flip_ab", "matched_flip_ab_ds3"]
ARMS = ["comp1", "random_norm_matched_squad2"]
DSETS = ["squad2", "umwp", "falseqa"]
JOB_FIELDS = ["exp", "split", "dataset", "condition", "pool", "sample_filter", "source_dataset",
              "base_vector_type", "arm", "layer", "alpha", "token_position", "scale_mode",
              "steering_mode", "position", "tag", "vector_type", "vector_source_dataset",
              "vector_npz"]


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sid_key(s):
    m = re.findall(r"\d+", s)
    return (int(m[-1]) if m else 0, s)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plan, jobs_by_model, samples_by_model = [], collections.defaultdict(list), collections.defaultdict(dict)
    gated = lambda r: r["epistemic_abstention"] and not r["is_degenerate"]

    for name, rid, mk in MODELS:
        # baseline verdicts: original pool plus, where it exists, the extension
        base, pool_rec = {}, {}
        for r in load(f"{PKG}/runs/twojudge/stance_peritem/{name}.jsonl"):
            if r["cell"] == "baseline_unans":
                base[(r["dataset"], r["sample_id"])] = gated(r)
        for r in load(f"{PKG}/runs/twojudge/stance_peritem_poolext/{name}.jsonl"):
            base[(r["dataset"], r["sample_id"])] = gated(r)
        for sub in ["matched_baseline", "matched_baseline_ext"]:
            for r in load(f"{PKG}/runs/{rid}/generations/{mk}/{sub}/generations.jsonl"):
                if r.get("should_abstain"):
                    pool_rec[(r["dataset"], str(r["sample_id"]))] = r

        tmpl, selected = {}, collections.defaultdict(set)
        for sub in FLIP_SUBS:
            for r in load(f"{PKG}/runs/{rid}/generations/{mk}/{sub}/generations.jsonl"):
                if r.get("arm") in ARMS and float(r.get("alpha", 0)) < 0:
                    tmpl.setdefault((r["dataset"], r["arm"]), {k: r[k] for k in JOB_FIELDS if k in r})
                    if r["arm"] == "comp1":
                        selected[r["dataset"]].add(str(r["sample_id"]))

        for ds in DSETS:
            sel = selected.get(ds)
            if not sel:
                continue
            drop = sorted([s for s in sel if not base.get((ds, s), False)], key=sid_key)
            if not drop:
                continue
            cand = sorted([s for (d, s) in base if d == ds and base[(d, s)] and s not in sel], key=sid_key)
            adds = cand[:len(drop)]
            if len(adds) < len(drop):
                print(f"  !! {name}|{ds}: candidates {len(adds)} < needed {len(drop)}")
                continue
            plan.append((name, ds, drop, adds))
            job_split = next((t["split"] for (d, a), t in tmpl.items() if d == ds), "matched_pool")
            for s in adds:
                rec = pool_rec.get((ds, s))
                if rec is None:
                    print(f"  !! {name}|{ds}: no sample record for {s}")
                    continue
                samples_by_model[mk][s] = {
                    "sample_id": rec["sample_id"], "dataset": rec["dataset"],
                    "split": job_split, "condition": rec.get("condition", "") or "",
                    "should_abstain": True,
                    "question": rec.get("question", ""), "prompt": rec.get("prompt", ""),
                    "reference_answers": rec.get("reference_answers", []),
                    "source_dataset": rec.get("source_dataset", ds)}
            for arm in ARMS:
                t = tmpl.get((ds, arm))
                if t is None:
                    print(f"  !! {name}|{ds}|{arm}: no frozen job template")
                    continue
                job = dict(t)
                job["model"] = mk
                job["sample_limit"] = len(adds)
                jobs_by_model[mk].append(job)

    print(f"{'model':22}{'jobs':>6}{'samples':>9}")
    for name, rid, mk in MODELS:
        if not jobs_by_model[mk]:
            continue
        with open(f"{OUTDIR}/{mk}.jobs.jsonl", "w") as f:
            for j in jobs_by_model[mk]:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        with open(f"{OUTDIR}/{mk}.samples.jsonl", "w") as f:
            for s in sorted(samples_by_model[mk], key=sid_key):
                f.write(json.dumps(samples_by_model[mk][s], ensure_ascii=False) + "\n")
        print(f"{mk:22}{len(jobs_by_model[mk]):>6}{len(samples_by_model[mk]):>9}")

    json.dump([{"model": n, "dataset": d, "drop": dr, "add": a} for n, d, dr, a in plan],
              open(f"{OUTDIR}/plan.json", "w"), indent=1)
    tot = sum(len(p[2]) for p in plan)
    print(f"\nwrote {OUTDIR}/  ({tot} items replaced -> {tot * 2} generations)")


if __name__ == "__main__":
    main()

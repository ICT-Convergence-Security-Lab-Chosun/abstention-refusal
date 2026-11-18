#!/usr/bin/env python3
"""Build steering jobs for the stance-selected induction set (replacement items only).

The induction matched set was selected with the LLM judge (gpt-4o-mini): keep answerable items the unsteered
model did not abstain on. Reporting now uses the stance judge, so selection has to use the same
judge or the reported baseline is not actually 0. 39 of the selected items turn out to be stance
abstentions at baseline; this replaces exactly those, drawing from the already-judged baseline pool
under the rule already used in runs/matched_replacement.json -- eligible, not already selected,
lowest sample_id first.

Job specs are copied field-for-field from the frozen matched_flip generations of the same
(model, dataset, arm) so layer, alpha, vector file and tag cannot drift; only the sample set
differs. Output goes to a separate directory so the frozen generations stay untouched.

Run: python3 src/analysis/build_v1sel_jobs.py
"""
import os, json, re, collections

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PURIFY = f"{PKG}/runs/twojudge/stance_matched_purify.json"
OUTDIR = f"{PKG}/runs/v1sel_jobs"
MODELS = [("Llama", "v2_full_llama31_8b", "llama31_8b_instruct"),
          ("Qwen35", "v2_full_qwen35_9b", "qwen35_9b"),
          ("Gemma3", "v2_full_gemma3_12b", "gemma3_12b"),
          ("Qwen", "v2_full_qwen", "qwen25_7b_instruct"),
          ("Gemma", "v2_full_gemma", "gemma2_9b_it")]
FLIP_SUBS = ["matched_flip_ab", "matched_flip_ab_ds3"]
ARMS = ["comp1", "random_norm_matched_squad2"]
# fields that define a job; everything else in a generation row is per-sample output
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
    pur = json.load(open(PURIFY))["cells"]
    samples_by_model = collections.defaultdict(dict)
    jobs_by_model = collections.defaultdict(list)
    plan = []

    for name, rid, mk in MODELS:
        # every frozen generation row, keyed by (dataset, arm) -> job template, and the sample
        # records we need for the replacement ids
        tmpl, pool_samples = {}, {}
        for sub in FLIP_SUBS:
            for r in load(f"{PKG}/runs/{rid}/generations/{mk}/{sub}/generations.jsonl"):
                if r.get("arm") in ARMS and float(r.get("alpha", 0)) > 0:
                    tmpl.setdefault((r["dataset"], r["arm"]), {k: r[k] for k in JOB_FIELDS if k in r})
        for r in load(f"{PKG}/runs/{rid}/generations/{mk}/matched_baseline/generations.jsonl"):
            pool_samples[(r["dataset"], str(r["sample_id"]))] = r

        for ds in ["squad2", "umwp", "falseqa", "trivia"]:
            cell = pur.get(f"{name}|{ds}")
            if not cell or cell.get("status") != "ok" or not cell["drop"]:
                continue
            need = len(cell["drop"])
            adds = cell["replacement_candidates"][:need]
            if len(adds) < need:
                print(f"  !! {name}|{ds}: candidates {len(adds)} < needed {need}")
                continue
            plan.append((name, ds, cell["drop"], adds))
            # sample_matches() keys on split/dataset/condition/should_abstain, and the
            # matched-pool items are not in splits/all_v2.jsonl, so the record is rebuilt from the
            # baseline generation and stamped with the split its job expects.
            job_split = next((t["split"] for (d, a), t in tmpl.items() if d == ds), "matched_pool")
            for sid in adds:
                rec = pool_samples.get((ds, str(sid)))
                if rec is None:
                    print(f"  !! {name}|{ds}: no sample record for {sid}")
                    continue
                samples_by_model[mk][str(sid)] = {
                    "sample_id": rec["sample_id"], "dataset": rec["dataset"],
                    "split": job_split, "condition": rec.get("condition", "") or "",
                    "should_abstain": bool(rec.get("should_abstain", False)),
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
        jp, sp = f"{OUTDIR}/{mk}.jobs.jsonl", f"{OUTDIR}/{mk}.samples.jsonl"
        with open(jp, "w") as f:
            for j in jobs_by_model[mk]:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        with open(sp, "w") as f:
            for sid in sorted(samples_by_model[mk], key=sid_key):
                f.write(json.dumps(samples_by_model[mk][sid], ensure_ascii=False) + "\n")
        print(f"{mk:22}{len(jobs_by_model[mk]):>6}{len(samples_by_model[mk]):>9}")

    json.dump([{"model": n, "dataset": d, "drop": dr, "add": a} for n, d, dr, a in plan],
              open(f"{OUTDIR}/plan.json", "w"), indent=1)
    print(f"\nwrote {OUTDIR}/  (plan.json: {sum(len(p[2]) for p in plan)} items replaced)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch the public source datasets and build the benchmark inputs the pipeline expects.

No benchmark text is shipped in this repo. This script downloads the public datasets and
writes them under data/ in the schema the pipeline reads:

  data/squad2_umwp_falseqa_rawq_normalized.jsonl   (squad2 + umwp + falseqa; answerable + unanswerable)
  data/advbench_harmful.jsonl                        (harmful prompts, for the refusal direction)
  data/alpaca_harmless.jsonl                         (benign prompts, contrast for the refusal direction)

Sources (see DATA_LICENSES.md): SQuAD 2.0, TriviaQA, AdvBench, Alpaca via HuggingFace;
UMWP and FalseQA via their public repositories. This reproduces the *method* on the public
data; it does not reproduce the paper's exact frozen item selection (01_prepare_splits then
draws cue/dev/test by seed). TriviaQA (answerable control) is fetched by 00 at eval time.

Run:  python3 scripts/prepare_data.py
"""
import os, sys, json, csv, io, urllib.request
from pathlib import Path

os.environ.setdefault("HF_HOME", os.environ.get("ABST_HF_CACHE", str(Path(__file__).resolve().parents[1] / "hf_cache")))
DATA = Path(os.environ.get("ABST_PKG", Path(__file__).resolve().parents[1])) / "data"
DATA.mkdir(parents=True, exist_ok=True)
PER = int(os.environ.get("PREPARE_PER_DATASET", "3000"))  # cap rows per benchmark dataset


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>6}  {path.name}")


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "prepare-data"}), timeout=timeout).read()


def build_squad2():
    from datasets import load_dataset
    d = load_dataset("rajpurkar/squad_v2", split="validation", revision="refs/convert/parquet")
    rows = []
    for i, r in enumerate(d):
        if i >= PER:
            break
        unans = len(r["answers"]["text"]) == 0
        q = (f"Respond to the question using only information given in the context.\n\n"
             f"Context: {r['context']}\n\nQuestion: {r['question']}")
        rows.append({"sample_id": f"Squad2Dataset:{i}", "dataset": "squad2", "dataset_class": "Squad2Dataset",
                     "source_index": i, "question": q,
                     "reference_answers_json": json.dumps(r["answers"]["text"]),
                     "should_abstain": bool(unans),
                     "scenario_label": "underspecified context" if unans else "answerable question"})
    return rows


def build_umwp():
    txt = get("https://raw.githubusercontent.com/Yuki-Asuuna/UMWP/main/data/StandardDataset.jsonl").decode()
    rows = []
    for line in txt.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["id"])
        if len(rows) >= PER:
            break
        unans = not bool(r.get("answerable", True))
        rows.append({"sample_id": f"UMWP:{i}", "dataset": "umwp", "dataset_class": "UMWP",
                     "source_index": i, "question": r["question"],
                     "reference_answers_json": json.dumps(r.get("answer")),
                     "should_abstain": bool(unans),
                     "scenario_label": "unanswerable math" if unans else "answerable question"})
    return rows


FALSEQA_URLS = [
    "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/exp-3/falseqa_test.csv",
    "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/exp-3/train_1187.csv",
    "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/exp-3/valid_1187.csv",
]


def build_falseqa():
    for url in FALSEQA_URLS:
        try:
            txt = get(url, timeout=30).decode()
        except Exception:
            continue
        rows = []
        for i, r in enumerate(csv.DictReader(io.StringIO(txt))):
            if i >= PER:
                break
            q = r.get("question") or r.get("Question") or next(iter(r.values()))
            fp_label = str(r.get("label", "1")).strip() == "1"   # 1 = false premise, 0 = true premise
            rows.append({"sample_id": f"FalseQADataset:{i}", "dataset": "falseqa", "dataset_class": "FalseQADataset",
                         "source_index": i, "question": q,
                         "reference_answers_json": json.dumps([r.get("answer", "")]),
                         "should_abstain": bool(fp_label),
                         "scenario_label": "false premise" if fp_label else "answerable question"})
        if rows:
            return rows
    print("  [warn] FalseQA could not be fetched automatically. Download dataset/test.csv from the\n"
          "         FalseQA repository (Hu et al., 2023) and place it at data/falseqa_test.csv, then\n"
          "         re-run; or proceed with squad2+umwp only (2 of the 3 extraction datasets).")
    local = DATA / "falseqa_test.csv"
    if local.exists():
        rows = []
        for i, r in enumerate(csv.DictReader(open(local, encoding="utf-8"))):
            q = r.get("question") or next(iter(r.values()))
            rows.append({"sample_id": f"FalseQADataset:{i}", "dataset": "falseqa", "dataset_class": "FalseQADataset",
                         "source_index": i, "question": q, "reference_answers_json": json.dumps([]),
                         "should_abstain": True, "scenario_label": "false premise"})
        return rows
    return []


def build_advbench():
    from datasets import load_dataset
    d = load_dataset("walledai/AdvBench", split="train")
    return [{"sample_id": f"advbench:{i}", "source_index": i, "question": r["prompt"]} for i, r in enumerate(d)]


def build_alpaca():
    from datasets import load_dataset
    d = load_dataset("tatsu-lab/alpaca", split="train")
    rows = []
    for i, r in enumerate(d):
        if r.get("input"):
            continue  # instruction-only, to match the benign contrast set
        rows.append({"sample_id": f"alpaca:{len(rows)}", "source_index": len(rows), "question": r["instruction"]})
        if len(rows) >= 600:
            break
    return rows


def main():
    print("Building benchmark inputs under data/ from public sources ...")
    bench = []
    for name, fn in [("squad2", build_squad2), ("umwp", build_umwp), ("falseqa", build_falseqa)]:
        try:
            r = fn(); bench += r; print(f"  {name:8} {len(r)} rows")
        except Exception as e:
            print(f"  [warn] {name} failed: {str(e)[:120]}")
    write_jsonl(bench, DATA / "squad2_umwp_falseqa_rawq_normalized.jsonl")
    for name, fn, out in [("advbench", build_advbench, "advbench_harmful.jsonl"),
                          ("alpaca", build_alpaca, "alpaca_harmless.jsonl")]:
        try:
            write_jsonl(fn(), DATA / out)
        except Exception as e:
            print(f"  [warn] {name} failed: {str(e)[:120]}")
    print("Done. Next: bash scripts/v2/00_generate_controlled_test.sh (TriviaQA control pool), then 01..10.")


if __name__ == "__main__":
    main()

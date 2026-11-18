# Data licenses, attribution, and responsible-use notice

No benchmark inputs or model outputs are shipped in this repository.
`scripts/prepare_data.py` downloads the public source datasets at run time and writes the
pipeline inputs under `data/` (git-ignored). Each dataset is used under its own license; the
code in this repo is MIT (`LICENSE`) and that grant does not extend to the fetched data.

## Source datasets

| Dataset | Used for | Source | License / terms |
|---|---|---|---|
| SQuAD 2.0 | answerable / unanswerable QA (`squad2`) | `rajpurkar/squad_v2` (HuggingFace) — https://rajpurkar.github.io/SQuAD-explorer/ | CC BY-SA 4.0 |
| UMWP | unanswerable math (`umwp`) | `Yuki-Asuuna/UMWP` — https://github.com/Yuki-Asuuna/UMWP | research use; attribute the UMWP paper |
| FalseQA | false-premise questions (`falseqa`) | `thunlp/FalseQA` — https://github.com/thunlp/FalseQA | research use; attribute the FalseQA paper |
| TriviaQA | answerable control (`trivia`) | `mandarjoshi/trivia_qa` (HuggingFace) — http://nlp.cs.washington.edu/triviaqa/ | Apache 2.0 / non-commercial research |
| AdvBench | reverse-steering safety eval (`advbench`) | `walledai/AdvBench` (HuggingFace) — https://github.com/llm-attacks/llm-attacks | MIT |
| Alpaca | benign contrast for the refusal direction | `tatsu-lab/alpaca` (HuggingFace) — https://github.com/tatsu-lab/stanford_alpaca | CC BY-NC 4.0 |

`prepare_data.py` pins each dataset's HuggingFace id / repository, so the fetched inputs are
traceable to the public source. Because the pipeline draws from the raw public data rather than
the paper's normalized pool, it reproduces the method and close results, not the identical
frozen selection.

## Responsible-use notice (harmful content, ARR A2 / B4)

The reverse-steering safety experiment (Section 4.3) applies AdvBench harmful prompts and records
the models' responses. When refusal is broken, the model produces unsafe outputs; these are
generated **locally at run time and are not included in this repository**. The reported safety
metric (validity-gated non-refusal) is computed from the WildGuard labels
(`src/analysis/wildguard_harm_check.py`) plus the rule-based validity check
(`src/analysis/compute_nonref_valid.py`).

- Run and use these outputs only in a research context; do not deploy or redistribute them.
- The AdvBench prompts are harmful by construction; the pipeline records model responses to them
  solely to measure a safety failure, not to endorse it.

## Personally identifying information (ARR B4)

The source datasets are public benchmarks and introduce no new personal data. Benchmark text may
mention public entities in factual questions (e.g. TriviaQA), but contains no private personal
information. This repository ships no data of its own.

# Abstention is not Refusal — Paper Code

Code and frozen artifacts to reproduce every reported number in the paper
*"Abstention is not Refusal, but it Affects Safety of Large Language Models."*

The release covers the experiments in the paper: geometry of the abstention vs refusal
directions, cross-task abstention induction/suppression, response frames, the
reverse-steering jailbreak, operating-point selection, and human validation.

## Two ways to use this release

The repository ships **code plus the frozen intermediate artifacts** under `runs/`.

- **Reproduce the paper numbers without a GPU or API** — every table and reported number
  is recomputed from the frozen per-item artifacts by a small analysis script (Section A).
- **Reproduce from scratch** — regenerate the artifacts with model weights (GPU) and an
  OpenAI-compatible API key for the LLM judge (Section B).

Paths resolve to the repository root automatically; optionally `export ABST_PKG=/path/to/repo`.
Run every command from the repository root.

## Setup

```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt
```

## A. Reproduce the paper numbers from frozen artifacts (no GPU, no API)

Each script reads the frozen per-item labels under `runs/` and prints the paper values.

```bash
export ABST_PKG=$(pwd)

# Table 2 / Fig 1,3 / §4.1  — geometry: cos(v_abs,v_ref), |cos|(cue_diff,v_abs), SVD PVE
python3 src/analysis/geometry_final.py

# Table 8 / Fig 5 / §4.3   — validity-gated non-refusal (+ 95% Wilson CIs)
python3 src/analysis/compute_nonref_valid.py

# Fig 4 / §4.2             — response-frame composition (epistemic / safety / other)
python3 src/analysis/compute_frames.py

# Table 5 c* (Stage B)     — comp1 dose-response operating point
python3 src/analysis/analyze_comp1_dose.py

# Table 5 L* (Stage A)     — norm-matched joint (layer x c) sweep
python3 src/analysis/analyze_joint_sweep.py

# Table 6 / §4.4           — human validation (agreement / kappa)
python3 src/analysis/score_human_audit_200.py --dir human_label_200
python3 human_label_200/score_audit_200.py    # self-contained scorer + provenance check
```

`analyze_comp1_dose.py` and `analyze_joint_sweep.py` are re-run safe: when the canonical
`*_summary.json` (the shipped paper value) is present, they write their recomputation to a
`*_summary.recomputed.json` sibling and print a per-model `MATCH` / `DIFF` comparison, never
overwriting the canonical file. Table 7 induction/suppression cells are read directly from
`runs/twojudge/stance_cells_v1sel.json` (+ `..._supp_v1sel.json`).

Artifact → paper-output map:

| Paper output | Frozen artifact (`runs/`) | Reproduce script |
|---|---|---|
| §4.1 geometry (Table 2, Fig 1/3) | `geometry_final.json`, `v2_full_*/vectors/*/v2_vectors.npz` | `src/analysis/geometry_final.py` |
| §4.2 induction/suppression (Table 7) | `twojudge/stance_cells_v1sel.json`, `..._supp_v1sel.json` | `src/analysis/judge_v1sel.py`, `judge_supp_v1sel.py` (from scratch) |
| §4.2 response frames (Fig 4) | `v2_full_*/generations/*/matched_flip_ab/…` | `src/analysis/compute_frames.py` |
| §4.3 non-refusal (Table 8, Fig 5) | `wildguard_heldout/wildguard_peritem_all.jsonl` | `src/analysis/compute_nonref_valid.py` |
| Table 5 L* (Stage A) | `exp_joint_sweep/joint_sweep_summary.json`, `twojudge/stance_peritem_joint_sweep/`, `hnorm_perlayer.json` | `src/analysis/analyze_joint_sweep.py` |
| Table 5 c* (Stage B) | `exp_comp1_dose/comp1_dose_summary.json`, `twojudge/stance_peritem_dose/` | `src/analysis/analyze_comp1_dose.py` |
| Fig 7 per-benchmark induction across layers | `twojudge/perds_ml.json` (frozen Stage-A LLM-judge grid) | from scratch: layer sweep judged by `src/analysis/ml_judge.py` |
| Table 6 human validation | `human_label_200/` | `src/analysis/score_human_audit_200.py` |
| Appendix-A balance (min cos 0.988 / 0.994) | `balance_check_all/cosines.json` | `src/analysis/balance_check_cosines.py` |

## B. Reproduce from scratch (GPU + model weights + API)

Provide the benchmark inputs under `data/` (public sources: SQuAD 2.0, UMWP, FalseQA,
AdvBench, Alpaca; see `configs/v2_defaults.env`) and an API key
(`cp .env.example .env.local`, fill `OPENROUTER_API_KEY`). Model weights are pulled by HF Hub
id (`src/steering/instruction_config.py`); a local snapshot is used when `HF_HOME`/`HF_HUB_CACHE`
point at it.

The exact item selection is pinned by text-free ID manifests under `provenance/`
(`cue_extract` 1500 / `dev` 700 / `test` 700 / `e7_refusal_samples` 100) — see
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for the `sample_id` → public-dataset mapping. Every
frozen artifact under `runs/` also carries `sample_id` for tracing.

Per-model core pipeline (once per model, with `RUN_ID` / `MODEL_KEY` overrides):

```bash
bash scripts/v2/00_generate_controlled_test.sh
bash scripts/v2/01_prepare_splits.sh
bash scripts/v2/02_extract_vectors_geometry.sh
bash scripts/v2/02b_extract_refusal_vectors.sh
bash scripts/v2/03_build_calibration_jobs.sh
bash scripts/v2/04_run_calibration_generation.sh
bash scripts/v2/05_judge_calibration.sh
bash scripts/v2/06_select_params.sh
bash scripts/v2/07_build_eval_jobs.sh
bash scripts/v2/08_run_eval_generation.sh
bash scripts/v2/09_judge_eval.sh
bash scripts/v2/10_summarize_plot.sh
```

Cross-model aggregations and the LLM-judge drivers (judge = `src/analysis/ml_judge.py`,
gpt-4o-mini):

- Geometry: `python3 src/analysis/geometry_final.py`
- Induction / suppression: `bash scripts/run_v1sel_generation.sh`, `run_supp_v1sel_generation.sh`,
  then `scripts/run_judge_v1sel.sh`, `run_judge_supp_v1sel.sh`
- Jailbreak: reverse-steering judged by `src/analysis/wildguard_harm_check.py` (WildGuard),
  Qwen-3.5 / Gemma-3 per-item via `scripts/reproduce/judge_decomp_heldout_q35.py`
- Operating point: Stage-A and Stage-B sweeps generated by `src/steering/run_v2_campaign.py`,
  judged into `twojudge/stance_peritem_joint_sweep/` and `twojudge/stance_peritem_dose/` by
  `src/analysis/judge_joint_sweep.py` and `judge_dose.py`, then `analyze_joint_sweep.py` /
  `analyze_comp1_dose.py`
- Balance: `src/analysis/balance_check_cosines.py`

## Human validation package (`human_label_200/`)

One author's blinded labels for the four automatic evaluators (200 items each, 100 per
validity stratum), the answer key (`KEY_200.json`), the scores, and a self-contained
scorer/verifier (`score_audit_200.py`). `build_human_audit_200.py` regenerates the blinded
sheets deterministically (`seed=20260804`) from the frozen generations; running the scorer
reproduces the Table 6 numbers (abstention 0.89 / κ 0.78, harmfulness 0.94 / κ 0.88,
refusal 0.95 / κ 0.90, frame 0.96 / κ 0.92, rule precision 0.86, miss-rate 0.01).

## Layout

```text
src/steering/    extraction, steering, campaign runner, selection
src/analysis/    geometry, LLM + WildGuard judges, sweep/dose analysis, number reproducers, human-audit scoring
scripts/v2/      00-10 per-model pipeline
scripts/         cross-model judge / generation drivers
human_label_200/ blinded human-validation package
provenance/      text-free sample_id manifests (exact splits)
runs/            frozen intermediate artifacts
```

## Verification

The GPU and API stages were executed with this release's code on local weights and checked
against the frozen artifacts:

- **Splits** — `01_prepare_splits.sh` reproduces the `cue_extract` set exactly (1500/1500 vs
  `provenance/`).
- **Geometry** — `02_extract_vectors_geometry.sh` reproduces `comp1` at every layer
  (|cos| to frozen = 0.999999, Llama).
- **Steered generation** — `run_v2_campaign.py` with the extracted `comp1` vector at
  (L*=13, c=0.65) drives Llama abstention on answerable SQuAD2 from 0.10 (unsteered) to 1.00,
  matching the reported induction.
- **Numbers** — every reported value (geometry, Table 5 headline L*/c*/α*, Tables 6/7/8 with
  all Wilson CIs, §4.2 frames, Appendix-A balance) recomputes from the frozen artifacts with
  no mismatch.

Known within-tolerance caveats (headline results unaffected; the shipped artifacts, which match
the paper, are canonical):

- **Selection judge, supplementary models.** Re-judging the Stage-A joint sweep with the LLM
  judge reproduces the deployed layer exactly for the headline Llama and Gemma-3 and one layer
  higher for Qwen-3.5 (19 vs 18, both on the induction plateau, Δ ≈ 0.03). The two supplementary
  models (Qwen-2.5, Gemma-2) show larger judge sensitivity in Stage A.
- **Gemma-2 dose (supplementary).** Re-running Stage-B on the LLM-judge labels returns c*=0.8 vs
  the deployed 0.65 (one grid step, plateau edge).
- **Qwen-3.5 TriviaQA induction.** The frozen value (0.91) uses a small item swap wired outside
  the released judge driver; re-judging from the shipped inputs gives 0.93.

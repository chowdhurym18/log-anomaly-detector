# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single, production-shaped experiment in **log anomaly detection** over the HDFS log dataset: a GRU learns `P(next EventId | sequence)`; sequences whose actual next event is "surprising" are scored as anomalies and routed to a local LLM for an evidence-grounded, anti-hallucination explanation.

**Post-audit reframing (the headline path).** The production pipeline is now a *suspicious-activity detector*: a GRU trained on **normal blocks only** (`run_normal_only.py`) scores how surprising each block is, validation-derived cutoffs sort blocks into **Normal / Uncertain / Suspicious**, and flagged blocks get the explanation. Run it with `venv/bin/python main.py --detect` → `outputs/detection_report.md`. This lifts held-out block-level detection **F1 from 0.21 (mixed training) to 0.72 (normal-only)** — see `docs/RESEARCH_POSITIONING.md`, `docs/ROADMAP.md`, and `outputs/normal_vs_anomaly_separation.md` / `outputs/detection_analysis.md`. The original next-event path (`main.py` with no flag, below) is unchanged and still powers the explanation layer and demo.

The code was refactored from one ~2,300-line `gru_pipeline.py` into a small package layout. **`main.py` is the entry point** (the `--detect` flag selects the detection pipeline). Behavior of the legacy next-event path is identical to the old monolith; only the file organization changed.

```
main.py                    entry point — argparse + orchestration only (~440 lines)
config.py                  CONFIG dict (single source of truth) + config_for_dataset() (HDFS/BGL registry) + get_device()
hdfs_v1_preprocess.py      one-time: Event_traces.csv → flat HDFS_v1_structured.csv
bgl_preprocess.py          one-time: raw BGL.log → structured CSV + traces (Drain3)
preprocessing/
  templates.py             EventId → Drain template loading / decoding
  encoder.py               StreamingLabelEncoder, cache validation, class weights
  dataset.py               build_memmap_sequences, MemmapLogDataset, build_dataloaders
  block_dataset.py         labelled per-block splits for the detection path (load_labeled_blocks)
  template_embeddings.py   semantic (template-text) embedding init (--semantic-embeddings)
models/gru_model.py        GRUAnomalyDetector (gru | bigru | bigru_attention) + AttentionPooling
training/train.py          train_model + _run_val_epoch (+ focal loss option)
evaluation/evaluate.py     evaluate_model, collect_anomaly_outputs, choose_thresholds, composite_score
evaluation/anomaly_eval.py ground-truth block-level grading (P/R/F1, PR-AUC, AUROC)
evaluation/attention.py    attention_table / format / CSV export (for the demo + report)
llm/ollama_client.py       chat handle, shared stats/cache/rate-limit, test_ollama_connection
llm/explanation.py         evidence-based report assembly + scrubbers + generate_explanation + send_to_llm
llm/uncertain_classifier.py constrained Llama Normal/Suspicious adjudicator for the Uncertain band
pipeline/anomaly_pipeline.py  run_anomaly_pipeline, run_quick_test, advanced_model (stub) — legacy
pipeline/detection_pipeline.py  run_detection_pipeline — PRODUCTION suspicious-activity detector (normal-only → surprise → Normal/Uncertain/Suspicious → explain); main.py --detect
pipeline/hybrid_pipeline.py   uncertainty-aware routing (--hybrid; only UNCERTAIN cases hit the LLM)
pipeline/uncertain_demo.py    the --demo-uncertain live cascade
utils/logging_utils.py     setup_logging()
utils/format.py            format_pct / band_label — human-readable % + "Suspicious" labels
utils/perf.py              timing helpers
compare_models.py          (top-level) trains/evaluates all 3 variants → comparison table
demo.py                    (top-level) single-sequence research-fair demo
demo_presentation.py       (top-level) guided end-to-end presentation demo
analyze_detection.py       (top-level) why normal-only wins: score separation + histograms → outputs/normal_vs_anomaly_separation.md
analyze_routing.py         (top-level) routing-efficiency analysis → outputs/routing_efficiency_report.md
baseline_hdfs.py           (top-level) classical per-block baselines → outputs/hdfs_baseline_report.md
run_normal_only.py         (top-level) mixed vs normal-only block training → normal_only_training_report.md + .checkpoints/normal_only/
run_scoring_comparison.py  (top-level) which surprise score best separates anomalies → outputs/detection_analysis.md
run_bgl_evaluation.py      (top-level) full BGL evaluation incl. chronological split → outputs/bgl_results.md, bgl_vs_hdfs.md
evaluate_llm_ladder.py     (top-level) LLM-explainer prompt ladder → outputs/llm_ladder_{hdfs,bgl}.csv
evaluate_uncertain.py      (top-level) Llama accuracy on the Uncertain band → outputs/llama_uncertain_evaluation.md
tests/test_explanation.py  plain-assert regression suite for the explanation layer (14 tests)
tests/test_models.py       plain-assert regression suite for the model variants + attention (10 tests)
docs/README.md             docs index — read in order
docs/ARCHITECTURE.md       module map + data flow + canonical terminology
docs/RESEARCH_POSITIONING.md  honest positioning vs DeepLog/LogAnomaly/LogBERT/LogGPT
docs/ROADMAP.md            future work ranked by evidence
docs/RESEARCH_AUDIT.md     the original audit that motivated normal-only training (historical)
docs/POSTER.md             research-fair poster copy
```

There is **no `finetuning/` directory** and there are **no `*_test.py` scratch demos** (`gru_model.py`, `gru_prep.py`, `dataset_test.py`, `llm_test.py` were removed). `tests/test_explanation.py` is a real, runnable test module (plain `assert`, no pytest).

## Environment & commands

Python **3.9.6** in a local `venv/`. Deps in `requirements.txt`: `torch 2.8`, `pandas`, `scikit-learn`, `numpy 2.0`, `scipy`, `matplotlib`, `ollama`, `drain3` (BGL parsing only). (No transformers/peft/accelerate — those were for the removed LoRA module.)

```bash
# Step 0 (one-time): convert HDFS_v1 Event_traces.csv → flat EventId CSV.
# Safe to re-run: exits immediately if output already exists.
venv/bin/python hdfs_v1_preprocess.py

# Step 1: delete stale cache whenever data_path or sequence_length changes.
rm -rf .gru_cache/ .checkpoints/

# Step 2: run the full pipeline (preprocess → train → evaluate → anomaly routing).
# MUST be run from the repo root — it uses relative paths like "data/...".
venv/bin/python main.py

# Fast inference-only paths (use the existing cache + .checkpoints/best_model.pt):
venv/bin/python main.py --eval-only
venv/bin/python main.py --quick-test --quick-test-n 20    # writes outputs/quick_test_results.csv

# Suspicious-activity detection — the POST-AUDIT PRODUCTION pipeline (normal-only GRU →
# surprise → Normal/Uncertain/Suspicious). Needs the encoder cache + the normal-only
# checkpoint (.checkpoints/normal_only/normal_only/best_model.pt from run_normal_only.py).
venv/bin/python run_normal_only.py                        # train mixed + normal-only detectors (block-level)
venv/bin/python main.py --detect                          # → outputs/detection_report.md (Normal/Uncertain/Suspicious)
venv/bin/python analyze_detection.py                      # why normal-only wins → outputs/normal_vs_anomaly_separation.md
venv/bin/python run_scoring_comparison.py                 # best surprise score → outputs/detection_analysis.md

# Model variants (research-fair upgrade — gru is the default + the existing checkpoint):
venv/bin/python main.py --variant bigru                   # train a bidirectional GRU
venv/bin/python main.py --variant bigru_attention         # BiGRU + attention pooling
venv/bin/python compare_models.py --reuse-existing        # 3-way table → outputs/model_comparison.{csv,md}
venv/bin/python demo.py                                    # single-sequence demo (attention + explanation)
venv/bin/python demo_presentation.py                      # guided end-to-end presentation demo

# BGL second-dataset validation (needs data/BGL/BGL.log downloaded + drain3 installed):
venv/bin/python bgl_preprocess.py                         # raw BGL.log → structured CSV + traces
venv/bin/python main.py --dataset bgl --detect            # BGL detector → outputs/bgl/detection_report.md
venv/bin/python run_bgl_evaluation.py                     # full BGL eval incl. chronological split

# Syntax-check every module without running:
python3 -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('**/*.py',recursive=True) if 'venv' not in f]; print('Syntax OK')"

# Run the regression suites (all tests):
venv/bin/python tests/test_explanation.py
venv/bin/python tests/test_models.py

# Run a SINGLE test (plain-assert functions — no pytest; call one by name):
venv/bin/python -c "import tests.test_explanation as t; t.test_root_cause_two_case_rule()"
```

There is no build step or linter config. "Verifying a change" means: syntax-check, run `tests/test_explanation.py`, and run `main.py --quick-test` (fast, uses the cache + saved model — no retrain).

### Ollama dependency
The LLM explanation step needs a local **Ollama** server running with the `llama3.2:1b` model pulled. The pipeline degrades gracefully if Ollama is absent: it still emits deterministic, template-grounded evidence reports (only the single LLM narrative field falls back to a Python default). So it runs fine without Ollama.

## Architecture

Driven entirely by the **`CONFIG` dict in `config.py`** — change behavior there, not by editing function internals. Every module does `from config import CONFIG`. Flow:

```
CSV → StreamingLabelEncoder (pass 1: vocab) → build_memmap_sequences (pass 2: sliding windows → disk)
    → MemmapLogDataset → train/val/test DataLoaders → GRUAnomalyDetector → train_model
    → evaluate_model → run_anomaly_pipeline → {send_to_llm | advanced_model}
```

Dependency direction is acyclic: `main → pipeline → {evaluation, llm.explanation} → {preprocessing, models, config}`, and `llm.explanation → llm.ollama_client` (never the reverse — that is what keeps the LLM package cycle-free). See `docs/ARCHITECTURE.md`.

- **Two-pass streaming design is deliberate and load-bearing.** Written to scale to 100M+ row logs even though smoke-tested on 2k rows. Pass 1 (`preprocessing/encoder.py`) builds the vocab with only a `Counter` in RAM; pass 2 (`preprocessing/dataset.py`) writes overlapping `(window, label)` pairs straight to memory-mapped `.npy` files via `np.memmap`. Nothing ever holds the full dataset in memory. Preserve this — don't introduce `pd.read_csv(whole_file)` or `np.array(all_sequences)`.
- **Disk cache in `.gru_cache/`** (`encoder.pkl`, `X.npy`, `y.npy`, `meta.pkl`). On startup, `main()` reuses it if `cache_is_valid()` passes (checks vocab size, `seq_len`, sequence count, validates index ranges). **If you change preprocessing logic, delete `.gru_cache/`** — stale caches pairing old arrays with a new encoder are the documented crash mode this validation guards against.
- **Checkpoints in `.checkpoints/`**: every epoch as `epoch_NNN.pt`, plus `best_model.pt` (best val_loss). `train_model` reloads `best_model.pt` at the end.
- **Model** (`models/gru_model.py`): `Embedding → LayerNorm → (Bi)GRU×N → [Attention] → LayerNorm → Dropout → Linear`. One class, three variants via `bidirectional`/`use_attention` flags, selected by `CONFIG["model_variant"]` or `--variant {gru,bigru,bigru_attention}`. **Defaults reproduce the original `gru` exactly, so `.checkpoints/best_model.pt` still loads.** Per-variant checkpoints: `gru → .checkpoints/`, others → `.checkpoints/<variant>/`. Device auto-selects CUDA → MPS → CPU via `get_device()`.
  - **Bidirectionality is SAFE here, not a leak** (the old comment was wrong): the target `events[i+seq_len]` lies *outside* the input window `events[i:i+seq_len]`, so reading the window both ways never sees the label. This only leaks in per-timestep next-token setups; this model makes one prediction past a fixed window.
  - **Attention** (`AttentionPooling`): learns one weight per timestep, softmaxes, returns the weighted average + the weights. The weights feed `demo.py` and the optional Python-owned "Model Attention Focus" section of the evidence report (`generate_explanation(..., attention=...)`). Changing the model does **not** invalidate `.gru_cache/` (data is model-independent) — only `.checkpoints/` is per-variant.

### Non-obvious design decisions (read the CONFIG comments before "fixing" these)

- **Current model size is `embedding_dim=64`, `hidden_dim=128` (~179K params)** for HDFS_v1 (29 EventIds, 8.9M training sequences) — conservative at 0.02× the training set, no collapse risk. The smaller `emb=32/hid=64` (~44K) was a *HDFS_2k* constraint to prevent majority-class memorization; that constraint no longer applies at HDFS_v1 scale.
- **Class weights use sqrt-inverse-frequency, NOT effective-number-of-samples.** The effective-number formula gave rare and common classes equal total gradient signal and made the model collapse to always predicting the rarest class (~0.25% acc). Documented at length in `compute_class_weights` (`preprocessing/encoder.py`).
- **Validation loss is intentionally *unweighted*** (`_run_val_epoch` in `training/train.py`) even though training loss is weighted — otherwise early stopping never fires when the model collapses to a rare class.
- **`label_smoothing` is disabled** (0.0) on purpose; combined with aggressive class weights it caused collapse.
- **Anomaly score is composite**, not pure surprisal: `surprisal·sw + entropy·ew + topk_miss·gw`. Entropy and top-K-miss were added because a class-biased model scores every non-majority actual as anomalous under pure surprisal. See `collect_anomaly_outputs` (`evaluation/evaluate.py`).
- **Thresholds default to `"calibrated"` mode** — anomaly/uncertain cutoffs are quantiles of the *training* score distribution (`anomaly_quantile`, `uncertain_quantile`), not the fixed `anomaly_threshold`. (Quick-test uses the fixed threshold to skip the train-scoring pass.)

### Evidence-based LLM layer (`llm/explanation.py`)
Hallucination is made structurally impossible: **Python owns every verdict field** (Expected/Observed event, Prediction Correct, Confidence/Anomaly Score, Classification, Root Cause Assessment, Recommended Investigation, Observed Sequence Pattern). The LLM is confined to ONE narrative field — `EVIDENCE_BASED_INTERPRETATION` — which is then triple-scrubbed: verdict restatements → speculation language → banned infrastructure causes not present verbatim in the involved Drain templates. Root Cause Assessment follows a strict two-case rule (`_root_cause_assessment`). Facts are wrapped in `BEGIN_FACTS/END_FACTS`. Calls are SHA-256-cached by case fingerprint, rate-limited, and retried with backoff; stats tracked thread-safely in `_llm_stats` (`llm/ollama_client.py`). `tests/test_explanation.py` regression-tests the whole scrubbing pipeline.

## Data

The pipeline is configured for HDFS_v1 (11.2M events, 29 EventIds, 575K blocks). HDFS_2k is kept for quick smoke-testing.

- [data/HDFS_v1/HDFS_v1_structured.csv](data/HDFS_v1/HDFS_v1_structured.csv) — **active training data** (generated by `hdfs_v1_preprocess.py`). Flat one-EventId-per-row CSV, ~11.2M rows, ~45 MB. Does not exist until you run the preprocess step.
- [data/HDFS_v1/preprocessed/Event_traces.csv](data/HDFS_v1/preprocessed/Event_traces.csv) — source for the above. 575K blocks, each with its full EventId sequence in the `Features` column (`[E5,E22,...]`). **Do not parse raw `HDFS.log`** — this file is already the parsed form.
- [data/HDFS_v1/preprocessed/anomaly_label.csv](data/HDFS_v1/preprocessed/anomaly_label.csv) — `BlockId → Normal/Anomaly` ground truth for 575K blocks. **Used by the detection path** (`preprocessing/block_dataset.load_labeled_blocks`) to build the labelled block split, derive the validation thresholds, and grade held-out detection. The detector still never sees labels at inference time — they only set thresholds on validation and score the test set.
- [data/HDFS_v1/preprocessed/HDFS.log_templates.csv](data/HDFS_v1/preprocessed/HDFS.log_templates.csv) — `EventId → EventTemplate` for all 29 event types. Loaded by `preprocessing/templates.load_event_templates()` and passed as `event_context` into the explanation layer.
- [data/HDFS_2k.log_structured.csv](data/HDFS_2k.log_structured.csv) — 2k-row flat CSV (same format). To switch back: change `data_path` in CONFIG, delete `.gru_cache/`, and reset model hyperparams to the 2K values.
- [data/BGL/](data/BGL/) — second-dataset validation (BlueGene/L supercomputer logs, per-message alert labels). Raw `BGL.log` is downloaded (see `data/BGL/README.md`), then `bgl_preprocess.py` derives `BGL_structured.csv`, `Event_traces.csv`, `anomaly_label.csv`, `BGL_templates.csv`. Wired via `config_for_dataset("bgl")`; caches to `.gru_cache_bgl/`, checkpoints to `.checkpoints/bgl/`.

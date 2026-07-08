# GRU-Based Log Anomaly Detection

A production-shaped, memory-efficient suspicious-activity detector for distributed system logs (HDFS, validated on BGL). A GRU trained on **normal blocks only** learns the normal conditional distribution of log events; blocks whose events are statistically surprising are sorted into **Normal / Uncertain / Suspicious** bands, and flagged blocks are routed to a local LLM for evidence-grounded, anti-hallucination reporting. Normal-only training lifts held-out block-level detection **F1 from 0.21 to 0.72** over the mixed-training baseline.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Anomaly Detection Methodology](#anomaly-detection-methodology)
- [Explanation Generation System](#explanation-generation-system)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This project detects anomalies in HDFS (Hadoop Distributed File System) log streams by framing the problem as next-event prediction: a GRU learns `P(next_event | history)`, and events assigned low probability are surprising. There are two entry paths:

- **Production detector** (`main.py --detect`) — the headline path. A GRU trained on **normal blocks only** (`run_normal_only.py`) scores each block's surprise; validation-derived cutoffs sort blocks into **Normal / Uncertain / Suspicious**; Suspicious blocks get evidence-grounded explanations and Uncertain blocks are adjudicated by a constrained Llama classifier.
- **Legacy next-event pipeline** (`main.py` with no flag) — trains on all sequences (normal + anomalous) and scores individual sliding windows with a composite surprise score. It still powers the explanation layer and the demos.

The same detector generalises to a second dataset, BGL supercomputer logs (`main.py --dataset bgl --detect`). The design scales from the 2 k-row HDFS_2k smoke-test dataset to the full 11.2 M-event HDFS_v1 dataset without holding the corpus in RAM.

The codebase is organised into single-responsibility packages (`preprocessing/`, `models/`, `training/`, `evaluation/`, `llm/`, `pipeline/`) orchestrated by a thin `main.py`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map and data flow.

---

## Features

- **Two-pass streaming preprocessing** — vocabulary built with a single `Counter` pass; sequences written to memory-mapped NumPy arrays via `np.memmap`. Never loads the full dataset into RAM.
- **Disk-backed sequence cache** — `.gru_cache/` stores the encoder, `X.npy`, `y.npy`, and metadata. Automatically invalidated when `sequence_length` or the source data changes.
- **Composite anomaly score** — three complementary signals: surprisal `1 − P(true_next)`, normalised Shannon entropy, and a top-K miss penalty. Mitigates class-bias false positives that pure surprisal produces.
- **Calibrated thresholds** — anomaly and uncertain cutoffs are computed as quantiles of the training-set score distribution, not hardcoded constants.
- **Sqrt-inverse-frequency class weights** — prevents majority-class collapse while keeping the gradient signal proportional to class frequency. Validated against the effective-number-of-samples alternative, which caused 0.25% accuracy on the imbalanced HDFS_2k set.
- **Unweighted validation loss** — guards early stopping against silent collapse to rare-class prediction when class weights are in use.
- **Evidence-based LLM explanations** — anomalies are forwarded to a local Ollama `llama3.2:1b` model via a multi-layer anti-hallucination architecture: structured facts boundary (`BEGIN_FACTS / END_FACTS`), three scrubbing stages (verdict restatements → speculation language → banned cause terms), and a Python-deterministic root-cause rule. All verdict fields are computed in Python and cannot be overridden by the model.
- **SHA-256 response cache** — avoids redundant Ollama calls for repeated anomaly patterns.
- **Hardware auto-selection** — CUDA → Apple MPS → CPU, with `pin_memory` and `prefetch_factor` tuned per device.
- **CLI flags** — `--detect` for the production Normal/Uncertain/Suspicious detector; `--dataset bgl` to run it on BGL; `--eval-only` to skip training and load a saved checkpoint; `--quick-test` / `--quick-test-n N` for fast subset evaluation with CSV output; `--experiment` for a lightweight subset run that never touches the production checkpoints.
- **Modular package layout** — config, preprocessing, model, training, evaluation, LLM, and pipeline orchestration are split into focused packages with an acyclic dependency graph (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

---

## Dataset

### HDFS_v1 (Primary)

| Property | Value |
|---|---|
| Source | Hadoop Distributed File System production logs |
| Blocks | 575,061 unique BlockIds |
| Total events | ~11.2 million |
| Unique EventIds | 29 (E1–E29) |
| Flat CSV size | ~45 MB |
| Ground truth | `anomaly_label.csv` — per-block Normal / Anomaly labels |

The raw dataset is distributed by [Loghub](https://github.com/logpai/loghub). `hdfs_v1_preprocess.py` converts the per-block `Event_traces.csv` (one bracketed sequence per row) into the flat one-EventId-per-row format consumed by the pipeline.

### HDFS_2k (Smoke-test)

A 2 k-row subset using the same column schema. Useful for rapid iteration; switch back by changing `data_path` in `CONFIG` and deleting `.gru_cache/`.

### BGL (Second-dataset validation)

BGL BlueGene/L supercomputer logs (4.7 M messages, per-message alert labels) validate that the detector is not HDFS-specific. See `data/BGL/README.md` for download and preprocessing steps (`bgl_preprocess.py` parses the raw log with Drain3). Key caveat documented in `outputs/bgl_vs_hdfs.md`: BGL detection F1 is 0.917 under a random split but 0.280 under a deployment-realistic chronological split.

### Downloading the datasets

The large dataset files are **not** committed to this repository — only the small template tables, the HDFS_2k smoke-test sample, and the dataset READMEs ship with it. To reproduce the HDFS results:

1. Download `HDFS_v1.zip` from Loghub ([Zenodo record 8196385](https://zenodo.org/records/8196385), also linked from [github.com/logpai/loghub](https://github.com/logpai/loghub)).
2. Extract `preprocessed/Event_traces.csv` and `preprocessed/anomaly_label.csv` into `data/HDFS_v1/preprocessed/`.
3. Run `venv/bin/python hdfs_v1_preprocess.py` to generate the flat `data/HDFS_v1/HDFS_v1_structured.csv`.

For BGL, follow `data/BGL/README.md` (download `BGL.log`, then run `bgl_preprocess.py`).

### Files Used

| File | Purpose | In repo? |
|---|---|---|
| `data/HDFS_v1/HDFS_v1_structured.csv` | Flat training data (generated by `hdfs_v1_preprocess.py`) | No — generated |
| `data/HDFS_v1/preprocessed/Event_traces.csv` | Source per-block sequences | No — download (step 2 above) |
| `data/HDFS_v1/preprocessed/anomaly_label.csv` | Ground-truth labels | No — download (step 2 above) |
| `data/HDFS_v1/preprocessed/HDFS.log_templates.csv` | EventId → Drain log template mappings | **Yes** |
| `data/HDFS_2k.log_structured.csv` | 2 k-row development subset | **Yes** |
| `data/BGL/BGL_templates.csv` | BGL EventId → Drain template table | **Yes** |

---

## Model Architecture

### GRUAnomalyDetector

```
Input (batch, seq_len=20)
  └─ Embedding(vocab=29, dim=64)
       └─ LayerNorm(64)            ← stabilises embedding gradient scale
            └─ GRU(64→128, layers=2, dropout=0.3)
                 └─ LayerNorm(128) ← stabilises hidden-state distribution
                      └─ Dropout(0.3)
                           └─ Linear(128 → 29)  [logits over EventIds]
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `embedding_dim` | 64 | Scaled for 29-class HDFS_v1; 32 was used for HDFS_2k to prevent memorisation |
| `hidden_dim` | 128 | ~179 K total params at 0.02× of 8.9 M training sequences |
| `num_layers` | 2 | Residual connections add value only at ≥ 4 layers |
| `dropout` | 0.3 | Applied between GRU layers and before the classifier head |
| `sequence_length` | 20 | Matches the average HDFS_v1 block length of 19.4 events |
| `label_smoothing` | 0.0 | Disabled — combined with class weights it causes collapse |

The baseline (`gru`) is unidirectional and summarises the window by its last timestep.

### Model variants (BiGRU + Attention)

One `GRUAnomalyDetector` class supports three architectures, selected by `--variant` (or `CONFIG["model_variant"]`). Defaults reproduce the original `gru` exactly, so the existing checkpoint still loads.

```
Events → Embedding → Bidirectional GRU → Attention → Classifier → next-event probabilities
```

| Variant | Reads window | Pooling | Params |
|---|---|---|---|
| `gru` (default) | left→right | last timestep | ~179K |
| `bigru` | both directions | last timestep | ~455K |
| `bigru_attention` | both directions | learned attention over all timesteps | ~456K |

**Bidirectionality is safe here (no leakage).** The predicted event `events[i+seq_len]` lies *outside* the input window `events[i:i+seq_len]`, so reading the window backward never sees the target. **Attention** learns a weight per event and returns a weighted average; those weights are exported (`outputs/attention_weights.csv`) and shown in the demo and explanation report ("these are the events the model focused on"). Compare all three with `python compare_models.py` (writes `outputs/model_comparison.{csv,md}`); run a single-sequence demo with `python demo.py`.

### Training

- **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-4`)
- **LR scheduler**: `ReduceLROnPlateau` on validation loss (`patience=3`)
- **Early stopping**: patience 7 epochs on unweighted validation loss
- **Class weights**: sqrt-inverse-frequency, normalised so mean weight = 1
- **Checkpointing**: every epoch to `.checkpoints/epoch_NNN.pt`; best checkpoint saved as `best_model.pt`

---

## Anomaly Detection Methodology

### Composite Anomaly Score

Each test sequence receives a score in [0, 1] combining three signals:

```
score = sw × surprisal + ew × entropy + gw × topk_miss
```

| Signal | Formula | What it catches |
|---|---|---|
| Surprisal | `1 − P(true_next_event)` | Model confidently predicted a different event |
| Entropy | `H(output) / log(vocab_size)` | Model is confused about what comes next |
| Top-K miss | `1` if true event ∉ top-3 predictions | True event is completely outside plausible continuations |

Weights: `entropy_weight=0.35`, `topk_miss_weight=0.10`. Surprisal takes the remainder.

### Calibrated Thresholds

> **Note (legacy path).** The percentile thresholds below belong to the *pre-audit* composite-score
> path (`main.py` with no flag). The **production** detector is `main.py --detect`, which ranks
> blocks by the detection score (`nll_-logp_mean`) and derives the Normal/Uncertain/Suspicious
> cutoffs from a **validation** set (`detection_pipeline.derive_bands`), not from training-score
> percentiles. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the canonical terminology.

Thresholds are computed from the training-set score distribution rather than hardcoded:

| Label | Condition |
|---|---|
| **ANOMALY** | score ≥ 95th percentile of training scores |
| **UNCERTAIN** | score ≥ 75th percentile of training scores |
| **NORMAL** | score < 75th percentile |

In this legacy path, ANOMALY sequences are routed to the LLM explanation layer and UNCERTAIN
sequences fall through to `advanced_model()`, a vestigial no-op stub. In the **production**
pipeline (`--detect`), Suspicious blocks are explained and **Uncertain blocks are adjudicated by
the constrained, scrubbed Llama classifier** (`llm/uncertain_classifier.py`) — there is no
"advanced model" stage.

---

## Explanation Generation System

Anomalous sequences are explained by a local Ollama instance running `llama3.2:1b`. The architecture is designed to make hallucination structurally impossible:

### Report Sections

| Section | Source |
|---|---|
| Summary — Observed Sequence Pattern | Python (`_default_observed_pattern`) |
| Prediction (Expected / Observed / Correct) | Python (deterministic comparison) |
| Confidence Score / Anomaly Score / Classification | Python (from detector) |
| Evidence-Based Interpretation | LLM (single scrubbed paragraph) |
| Root Cause Assessment | Python (strict two-case rule) |
| Recommended Investigation | Python (template-grounded steps) |

### Anti-Hallucination Layers

1. **Facts boundary** — all facts wrapped in `BEGIN_FACTS / END_FACTS`; the system prompt instructs the model that everything outside is unknown.
2. **Verdict scrubbing** — any LLM sentence that echoes "Confidence Score", "Prediction Correct", "Classification", etc., is removed before display.
3. **Speculation scrubbing** — sentences containing "suggests", "indicates", "likely", "possibly", "appears to", etc., are removed.
4. **Banned cause scrubbing** — sentences introducing infrastructure causes (disk, memory, network, CPU, hardware, storage, throughput, etc.) not present verbatim in the event templates are dropped.
5. **Python root-cause rule** — if the actual event's Drain template contains a failure token (`error`, `failed`, `failure`, `exception`, `timeout`, `denied`), the root cause reads: *"The observed event template contains `<token>`. The sequence does not provide enough evidence to determine a deeper root cause."* Otherwise: *"The sequence does not provide enough evidence to determine a root cause."*

### Drain Template Enrichment

`load_event_templates()` maps each EventId to its human-readable Drain log template (e.g., `E5` → `"Verification succeeded for <...>"`). The LLM receives these templates instead of opaque symbolic IDs, enabling factual, template-grounded narratives.

---

## Performance Metrics

### Headline result — block-level suspicious-activity detection

Training the detector on **normal blocks only** (instead of the mixed corpus) is the single change that made detection work: held-out block-level **F1 rose from 0.21 to 0.72** (peak-surprise rule; see `outputs/normal_vs_anomaly_separation.md`). The production detector (`main.py --detect`, graded against the handcrafted per-block labels it never sees at inference):

| Metric (HDFS_v1, held-out blocks) | Value |
|---|---|
| Suspicious-band F1 | **0.674** (Precision 0.802 · Recall 0.581) |
| PR-AUC (anomalies are 2.9% of blocks) | **0.692** |
| AUROC | **0.892** |

Full report: `outputs/detection_report.md`.

### Second dataset — BGL

| Metric | BGL (random split) | BGL (chronological split) |
|---|---|---|
| Detection F1 | **0.917** | **0.280** |
| PR-AUC | 0.950 | 0.140 |

The random-vs-chronological gap is a deliberately reported honesty caveat — random splits flatter BGL because failure modes repeat across time. See `outputs/bgl_vs_hdfs.md` and `docs/RESEARCH_POSITIONING.md`.

### Next-event proxy metrics (legacy path)

Evaluated on 2,235,122 held-out test sequences from the HDFS_v1 dataset (20% test split):

| Metric | Value |
|---|---|
| Top-1 Accuracy | **92.07%** |
| Top-3 Accuracy | **99.67%** |
| Weighted F1 | **0.9184** |

These measure how well the GRU models normal sequence structure — a proxy, not detection quality; the block-level numbers above are the ones that matter (`outputs/hdfs_baseline_report.md` explains the distinction).

---

## Project Structure

```
log-anomaly-detector/
│
├── main.py                    # Entry point — argparse + orchestration (--detect = production path)
├── config.py                  # CONFIG dict (single source of truth) + get_device()
├── hdfs_v1_preprocess.py      # One-time: Event_traces.csv → HDFS_v1_structured.csv
├── bgl_preprocess.py          # One-time: BGL.log → structured CSV + traces (Drain3)
│
├── run_normal_only.py         # Trains mixed vs normal-only block detectors (the headline experiment)
├── run_scoring_comparison.py  # Which surprise score best separates anomalies
├── run_bgl_evaluation.py      # Full BGL evaluation incl. chronological-split caveat
├── analyze_detection.py       # Why normal-only wins: score separation + histograms
├── analyze_routing.py         # Routing-efficiency analysis (Normal/Uncertain/Suspicious bands)
├── baseline_hdfs.py           # Classical baselines for context (per-block count models)
├── compare_models.py          # gru vs bigru vs bigru_attention comparison table
├── evaluate_llm_ladder.py     # LLM-explainer ladder evaluation (HDFS + BGL)
├── evaluate_uncertain.py      # Llama-adjudication accuracy on the Uncertain band
├── demo.py                    # Single-sequence research-fair demo (attention + explanation)
├── demo_presentation.py       # Guided end-to-end presentation demo
│
├── preprocessing/
│   ├── templates.py           # EventId → Drain template loading / decoding
│   ├── encoder.py             # StreamingLabelEncoder, cache validation, class weights
│   ├── dataset.py             # build_memmap_sequences, MemmapLogDataset, build_dataloaders
│   ├── block_dataset.py       # Labelled per-block splits for the detection path
│   └── template_embeddings.py # Semantic (template-text) embedding initialisation
├── models/
│   └── gru_model.py           # GRUAnomalyDetector (gru | bigru | bigru_attention) + attention pooling
├── training/
│   └── train.py               # train_model + validation loop (early stopping, LR schedule)
├── evaluation/
│   ├── evaluate.py            # evaluate_model, collect_anomaly_outputs, choose_thresholds
│   ├── anomaly_eval.py        # Ground-truth block-level grading (precision/recall/F1, PR-AUC)
│   └── attention.py           # Attention tables / CSV export for the demo + report
├── llm/
│   ├── ollama_client.py       # chat handle, shared stats/cache/rate-limit, connectivity probe
│   ├── explanation.py         # evidence-based report assembly + scrubbers + send_to_llm
│   └── uncertain_classifier.py# Constrained Llama Normal/Suspicious adjudicator for the Uncertain band
├── pipeline/
│   ├── detection_pipeline.py  # PRODUCTION suspicious-activity detector (main.py --detect)
│   ├── anomaly_pipeline.py    # Legacy next-event pipeline (run_anomaly_pipeline, run_quick_test)
│   ├── hybrid_pipeline.py     # Uncertainty-aware routing (only UNCERTAIN cases go to the LLM)
│   └── uncertain_demo.py      # Live two-stage cascade demo (--demo-uncertain)
├── utils/
│   ├── logging_utils.py       # setup_logging()
│   ├── format.py              # format_pct / band_label
│   └── perf.py                # Timing helpers
├── tests/
│   ├── test_explanation.py    # Plain-assert regression suite for the explanation layer (14 tests)
│   └── test_models.py         # Plain-assert regression suite for model variants + attention (10 tests)
├── docs/
│   ├── README.md              # Docs index — read in order
│   ├── ARCHITECTURE.md        # Module map + data flow + canonical terminology
│   ├── RESEARCH_POSITIONING.md# Honest positioning vs DeepLog / LogAnomaly / LogBERT / LogGPT
│   ├── ROADMAP.md             # Future work ranked by evidence
│   ├── RESEARCH_AUDIT.md      # The original audit that motivated normal-only training
│   └── POSTER.md              # Research-fair poster copy
│
├── data/
│   ├── HDFS_2k.log_structured.csv   # 2k-row smoke-test subset (committed)
│   ├── HDFS_2k.log_templates.csv    # Drain templates for HDFS_2k (committed)
│   ├── HDFS_v1/                     # Large files downloaded/generated — see "Downloading the datasets"
│   └── BGL/                         # BGL templates committed; raw log downloaded — see data/BGL/README.md
│
├── outputs/                   # Committed result reports (.md) + figures (.png) + small CSVs
├── .checkpoints/              # Trained models — the three production best_model.pt files are committed
├── .gru_cache/                # Local only: memmap sequence cache (regenerated from the dataset)
│
├── CLAUDE.md                  # Codebase guide for AI coding assistants
├── PROJECT_CLEANUP.md         # Curation record: what was kept/archived/superseded and why
├── final_verification.md      # End-state verification: checksums + all-workflows pass table
├── LICENSE                    # MIT (code); data/ files carry LogHub terms
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.9.6
- [Ollama](https://ollama.com/) (optional — for LLM explanations)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/chowdhurym18/log-anomaly-detector.git
cd log-anomaly-detector

# 2. Create and activate a virtual environment
python3.9 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pull the LLM for explanation generation
ollama pull llama3.2:1b

# 5. Download the dataset (see "Downloading the datasets" above):
#    put Event_traces.csv + anomaly_label.csv in data/HDFS_v1/preprocessed/

# 6. Preprocess HDFS_v1 (one-time — converts per-block CSV to flat format)
python hdfs_v1_preprocess.py
```

> **Note**: `hdfs_v1_preprocess.py` is idempotent — it exits immediately if the output file already exists.

---

## Usage

All commands are run from the repo root (the pipeline uses relative paths like `data/...`).

### Production suspicious-activity detector

```bash
venv/bin/python main.py --detect
```

Scores every block with the normal-only GRU, sorts blocks into **Normal / Uncertain / Suspicious** using validation-derived cutoffs, sends flagged blocks to the explanation layer, and writes `outputs/detection_report.md`. Uses the committed checkpoint `.checkpoints/normal_only/normal_only/best_model.pt` (retrain it from scratch with `venv/bin/python run_normal_only.py`). Run the same detector on BGL with `venv/bin/python main.py --dataset bgl --detect`.

### First run on a fresh clone (no retraining)

The three production checkpoints are committed, but the `.gru_cache/` encoder/sequence cache is not (it is dataset-derived). After downloading the dataset (Installation step 5), build the cache once **without touching the committed checkpoints**:

```bash
venv/bin/python main.py --experiment   # builds .gru_cache/; trains only a small subset model into .checkpoints/experiment/
venv/bin/python main.py --detect       # production detector, uses the committed normal-only checkpoint
venv/bin/python main.py --eval-only    # full held-out evaluation of the committed legacy model
```

### Full pipeline (preprocess → train → evaluate → anomaly detection)

```bash
venv/bin/python main.py
```

> This is the legacy next-event path; it retrains from scratch and overwrites `.checkpoints/best_model.pt`.

### Skip training — load saved checkpoint and evaluate

```bash
venv/bin/python main.py --eval-only
```

Requires `.checkpoints/best_model.pt` (committed) and a valid `.gru_cache/` directory (see "First run on a fresh clone").

### Fast subset evaluation (no LLM, saves CSV)

```bash
# Score 500 random test sequences (default)
venv/bin/python main.py --quick-test

# Score 2000 random test sequences
venv/bin/python main.py --quick-test --quick-test-n 2000
```

Output is written to `outputs/quick_test_results.csv`.

### Syntax check without running

```bash
python3 -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('**/*.py',recursive=True) if 'venv' not in f]; print('Syntax OK')"
```

### Run the regression tests

```bash
venv/bin/python tests/test_explanation.py   # 14 explanation-layer tests
venv/bin/python tests/test_models.py        # 10 model-variant/attention tests
```

### Demos and model comparison

```bash
venv/bin/python demo_presentation.py        # guided end-to-end walkthrough
venv/bin/python demo.py                     # single-sequence demo (attention + explanation)
venv/bin/python main.py --demo-uncertain    # live two-stage cascade on Uncertain blocks (needs Ollama)
venv/bin/python compare_models.py --reuse-existing   # gru vs bigru vs bigru_attention table
```

### Stale cache — when to delete

Delete `.gru_cache/` and `.checkpoints/` whenever:
- `data_path` in `CONFIG` changes
- `sequence_length` in `CONFIG` changes
- You modify the preprocessing logic

```bash
rm -rf .gru_cache/ .checkpoints/
```

---

## Example Output

### Quick-test console summary

```
====================================================================
  Quick-Test Results  (500 samples, fixed threshold=0.6)
====================================================================
  Top-1 Accuracy  : 92.07%  (460/500)
  Anomaly Score   : min=0.0003  mean=0.1842  max=0.9971
  ✅  Normal      :    374  (74.8%)
  ❓  Uncertain   :    101  (20.2%)
  ⚠   Anomaly     :     25  (5.0%)
--------------------------------------------------------------------
  Sample predictions (first 10):
  [✓] pred=E9    actual=E9    score=0.0010  [NOR]
  [✓] pred=E5    actual=E5    score=0.0208  [NOR]
  [✓] pred=E5    actual=E5    score=0.0033  [NOR]
  [✗] pred=E5    actual=E11   score=0.6723  [ANO]
====================================================================
```

### quick_test_results.csv columns

| Column | Description |
|---|---|
| `sequence` | Space-separated EventId window (length 20) |
| `predicted_event` | Top-1 model prediction |
| `actual_event` | True next EventId |
| `correct` | Whether prediction matched |
| `anomaly_score` | Composite score ∈ [0, 1] |
| `confidence` | `1 − anomaly_score` |
| `classification` | `NORMAL` / `UNCERTAIN` / `ANOMALY` |

### LLM explanation report (example)

```
Summary
  Observed Sequence Pattern: 20 events ending in E5 (Verification
  succeeded for <...>) -> E22 (Block <...> served to <...>) ->
  E5 (Verification succeeded for <...>) -> E11 (PacketResponder
  <...> for block <...> terminating) -> E9 (Receiving block <...>).

Prediction
  Expected Event    : E9 — Receiving block <...>
  Observed Event    : E11 — PacketResponder <...> for block <...> terminating
  Prediction Correct: False

Confidence
  Confidence Score  : 0.3277
  Anomaly Score     : 0.6723
  Classification    : ANOMALY

Evidence-Based Interpretation
  The expected event template describes initiating a block receive
  operation, while the actual event template describes a
  PacketResponder termination. These two templates represent
  different phases of block data transfer and do not correspond.

Root Cause Assessment
  The sequence does not provide enough evidence to determine a root cause.

Recommended Investigation
  * Review the raw HDFS log lines matching the observed event E11:
    "PacketResponder <...> for block <...> terminating".
  * Compare against the expected event E9: "Receiving block <...>".
    Confirm whether the E9 -> E11 transition is valid for this block.
  * Verify whether this exact event sequence has occurred in
    known-normal traffic for the same BlockId.
```

---

## Future Improvements

Ranked with evidence in [docs/ROADMAP.md](docs/ROADMAP.md). Highlights:

- **Chronologically-robust BGL detection** — the F1 0.917 → 0.280 collapse under a chronological split is the biggest open problem; concept-drift handling (periodic retraining, sliding reference windows) is the natural next step.
- **Transformer baseline** — compare the GRU against a small Transformer encoder (e.g., LogBERT or a custom BERT-Log) on the same block-level split.
- **LoRA-fine-tuned explainer** — the prompt-engineering ladder for the Llama adjudicator was exhausted without beating simple baselines (`outputs/llm_ladder_hdfs.csv`); fine-tuning is the evidence-gated next rung.
- **Online / incremental learning** — extend the streaming design to support continuous model updates as new log blocks arrive, without reprocessing the full corpus.
- **Pluggable LLM backends** — add `llm/openai_client.py` / `llm/anthropic_client.py` beside `ollama_client.py` behind a shared `explain()` signature.
- **Distributed training** — the memmap design is compatible with `torch.distributed`; add `DistributedSampler` support for multi-GPU training on the full HDFS_v1 corpus.

Completed items that used to live on this list: ground-truth block-level evaluation (`evaluation/anomaly_eval.py`), block-boundary-aware scoring (the `--detect` path scores whole blocks), and template-semantic embedding initialisation (`--semantic-embeddings`; flat vs the baseline — see `outputs/semantic_embedding_report.md`).

---

## Author

**Mohaimenul Hoque Chowdhury**
[mohaimenulhoquec@gmail.com](mailto:mohaimenulhoquec@gmail.com)

---

## References

- Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael Jordan. *Detecting Large-Scale System Problems by Mining Console Logs.* SOSP 2009. (HDFS dataset)
- Adam Oliner, Jon Stearley. *What Supercomputers Say: A Study of Five System Logs.* DSN 2007. (BGL dataset)
- Jieming Zhu et al. *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.* ISSRE 2023. (dataset distribution)
- He, Pinjia, et al. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* ICWS 2017.

The HDFS and BGL datasets are redistributed by [Loghub](https://github.com/logpai/loghub) for research purposes; see `data/HDFS_v1/README.md` and `data/BGL/README.md` for the required citations.

# Architecture


## Module map

```
config.py                  CONFIG dict (single source of truth) + get_device()
main.py                    Entry point: argparse + orchestration only
                           Flags: --eval-only, --quick-test, --variant,
                           --hybrid, --experiment, --max-train/--max-val/--max-test

preprocessing/
  templates.py             EventId → Drain template loading / decoding
  encoder.py               StreamingLabelEncoder, cache validation, class weights
  dataset.py               build_memmap_sequences, MemmapLogDataset, build_dataloaders
  template_embeddings.py   semantic event-embedding init from template text (TF-IDF+SVD)

models/
  gru_model.py             GRUAnomalyDetector — variants: gru | bigru | bigru_attention
                           (Embedding→LN→(Bi)GRU→[Attention]→LN→Dropout→Linear) + AttentionPooling

training/
  train.py                 train_model + _run_val_epoch (early stop, LR schedule)

evaluation/
  evaluate.py              evaluate_model, choose_thresholds, composite_score,
                           collect_anomaly_outputs (composite score),
                           collect_confidence_outputs (MSP confidence — hybrid),
                           collect_uncertainty_signals (MSP + top-2 gap + entropy — analysis)
  anomaly_eval.py          REAL block-level detection vs anomaly_label.csv:
                           P/R/F1/AUROC/PR-AUC + ROC/PR curves; detection_metrics() helper
  attention.py             attention_table / format / CSV export

llm/
  ollama_client.py         chat handle, shared stats/cache/rate-limit, connectivity probe
  explanation.py           evidence-based report assembly, scrubbers, generate_explanation, send_to_llm
                           (optional Python-owned "Model Attention Focus" section)

pipeline/
  anomaly_pipeline.py      run_anomaly_pipeline (composite-score, DEFAULT), run_quick_test, advanced_model (stub)
  hybrid_pipeline.py       run_hybrid_pipeline (confidence/MSP routing, only Uncertain → LLM),
                           classify_by_confidence (shared Normal/Uncertain/Anomaly banding rule)

utils/
  logging_utils.py         setup_logging()

demo_presentation.py       (top-level) THE live demo — one block per band, production path
main.py --detect           the PRODUCTION detector (pipeline/detection_pipeline.py)
run_normal_only.py         mixed-vs-normal-only block training → the detector checkpoint
run_scoring_comparison.py  which surprise score separates best → outputs/[bgl/]detection_analysis.md
run_bgl_evaluation.py      measured BGL + BGL-vs-HDFS reports (+ chronological-split check)
evaluate_llm_ladder.py     Llama uncertain-band improvement ladder → outputs/llama_bgl_evaluation.md
evaluate_uncertain.py      can Llama adjudicate the uncertain band? → llama_uncertain_evaluation.md
compare_models.py          train/evaluate all 3 variants → outputs/model_comparison.{csv,md}
demo.py                    single-sequence attention demo (LEGACY composite-score path)
# Superseded/one-off scripts (run_generalization, evaluate_hybrid, evaluate_explanations,
# compare_training, analyze_results, run_final_comparison, analyze_attention) moved to
# archive/scripts/ — see PROJECT_CLEANUP.md.

tests/
  test_explanation.py      plain-assert regression suite for the explanation layer
  test_models.py           plain-assert regression suite for the model variants + attention
```

## Data flow

```
CSV (data/HDFS_v1/HDFS_v1_structured.csv)
  │
  ├─ pass 1: StreamingLabelEncoder.fit ─────────────► vocabulary (Counter only in RAM)
  │
  ├─ pass 2: build_memmap_sequences ────────────────► .gru_cache/X.npy, y.npy (np.memmap)
  │
  ├─ build_dataloaders ─────────────────────────────► train / val / test (index split only)
  │
  ├─ GRUAnomalyDetector + train_model ──────────────► .checkpoints/best_model.pt
  │
  ├─ evaluate_model ────────────────────────────────► top-1/3/5, per-class F1, confusion
  │
  └─ routing (two interchangeable paths; main.py picks one)
        │
        ├─ run_anomaly_pipeline            ◄── DEFAULT (plain `main.py`)
        │     ├─ collect_anomaly_outputs (composite score) + choose_thresholds (calibrated)
        │     ├─ classify: Normal / Uncertain / Anomaly
        │     ├─ Anomaly   → send_to_llm → generate_explanation (Ollama, scrubbed)
        │     └─ Uncertain → advanced_model (stub)
        │
        └─ run_hybrid_pipeline             ◄── OPT-IN (`main.py --hybrid` / `--experiment`)
              ├─ collect_confidence_outputs  (confidence = max softmax prob, "MSP")
              ├─ classify_by_confidence: ≥0.70 Normal / ≤0.40 Anomaly / else Uncertain
              ├─ Uncertain ONLY → send_to_llm → generate_explanation (Ollama, scrubbed)
              └─ outputs/hybrid_results.csv
```

The two routing paths are independent and backward compatible: plain `main.py`
still runs the composite-score `run_anomaly_pipeline`. The hybrid path is the
diagrammed flow — *HDFS logs → templates → GRU → confidence → Normal/Uncertain/
Anomaly → only Uncertain → Llama → evidence-based explanation* — and is selected
with `--hybrid` (or `--experiment`, which also trains/evaluates on a small subset
and isolates checkpoints under `.checkpoints/experiment/`).

## Dependency direction (acyclic)

```
main ──► pipeline ──► evaluation ──► preprocessing ──► config
   │         │            │                              ▲
   │         └─► llm.explanation ─► llm.ollama_client ───┘
   │
   └─► training ─► models / preprocessing
```

`llm.explanation` imports `llm.ollama_client` (shared state + the `chat` handle),
never the reverse — this is what keeps the LLM package free of import cycles.

## Terminology (canonical — read this first)

These words mean specific things throughout the project. The **production** detector
(`main.py --detect`) uses these definitions; two legacy paths use *different* ones (flagged
below) — do not conflate them.

- **Block** — one unit that gets a verdict: an HDFS operation (a `BlockId`) or a fixed
  100-line BGL window. **Window** — one `(context → next-event)` pair inside a block.
- **Confidence score** — P(the *actual* next event) = 1 − anomaly score. High = the model
  expected what happened.
- **Anomaly score** — per-window **surprisal** = 1 − P(actual event), in [0, 1]. High = surprising.
- **Detection score** — the block-level surprise aggregate that the bands are derived from:
  `nll_-logp_mean` (mean per-window negative-log-likelihood). This is what ranks blocks.
- **Bands** — **Normal / Uncertain / Suspicious** (presentation names). The internal enum is
  `NORMAL / UNCERTAIN / ANOMALY`; `ANOMALY → "Suspicious"` via `utils/format.band_label`. Both
  band cutoffs are learned on validation (`detection_pipeline.derive_bands`), never hand-picked.
- **MSP (Maximum Softmax Probability)** — P(the model's *top-1 predicted* event). A **different**
  quantity from "confidence" above. Only the *legacy* hybrid path routes on MSP; it is the
  **worst** block-level detector (see `outputs/detection_analysis.md`), which is why production
  routes surprise, not MSP.

> Two legacy definitions to avoid: the **composite anomaly score** (`sw·surprisal + ew·entropy +
> gw·topk_miss`, below) belongs to the pre-audit `run_anomaly_pipeline`, and **confidence = MSP**
> belongs to the `--hybrid` path. Reports that used those (`hybrid_evaluation.md`,
> `explanation_quality_report.md`) are retained in `outputs/` for the audit trail but marked
> SUPERSEDED in PROJECT_CLEANUP.md — do not read them for current numbers.

## Design properties preserved from the original

- **Two-pass streaming** — nothing ever loads the full dataset into RAM; scales
  to 100M+ rows. Do not introduce `pd.read_csv(whole_file)` or `np.array(all)`.
- **Validated disk cache** (`.gru_cache/`) — `cache_is_valid()` guards against
  pairing stale arrays with a new encoder. Delete `.gru_cache/` when preprocessing
  logic or `sequence_length` changes.
- **Composite anomaly score** — `sw·surprisal + ew·entropy + gw·topk_miss`, not
  pure surprisal (a class-biased model would flag every non-majority event).
- **Calibrated thresholds** — quantiles of the *training* score distribution
  (composite-score path only).
- **Confidence / MSP routing (hybrid path)** — uses the model's own
  Maximum Softmax Probability (the probability of its top predicted next event),
  with fixed cutoffs `confidence_normal_threshold` (0.70) and
  `confidence_anomaly_threshold` (0.40). Only the **uncertain middle band** is
  escalated to the LLM; confident-normal is auto-accepted and confident-low is
  auto-flagged, so the expensive LLM is spent only where the model is genuinely
  unsure. `classify_by_confidence` is the single source of truth for the banding.
- **Experiment-checkpoint isolation** — `--experiment` trains/evaluates on a
  small subset (`experiment_max_train/val/test`) and writes to
  `.checkpoints/experiment/…`, never overwriting the production `best_model.pt`.
- **Evidence-based explanations** — Python owns every verdict field; the LLM is
  confined to one narrative field that is triple-scrubbed (verdict / speculation /
  banned-cause). See `llm/explanation.py`. Shared LLM stats
  (`llm/ollama_client.py`) track calls, cache hits, **failures, and latency**.

## Post-audit additions (detection, imbalance, semantics, multi-dataset)

All additive and flag-gated — the production `gru` checkpoint and behaviour are
unchanged (see `docs/RESEARCH_AUDIT.md` for the motivation, `outputs/final_summary.md`
for the measured results).

- **Real anomaly-detection metric** (`evaluation/anomaly_eval.py`) — re-windows each
  block from `Event_traces.csv` (no cross-block leakage), scores it with the GRU,
  and reports block-level P/R/F1/AUROC/PR-AUC vs `anomaly_label.csv`. This is the
  paper-comparable metric the project previously lacked. **Key finding:** the GRU,
  trained on *all* blocks (normal + anomalous), is a weak detector — DeepLog-style
  detection needs *normal-only* training.
- **Focal loss** (`CONFIG["loss"] = "focal"`, `--loss focal`) — `training/train.py`
  `FocalLoss` down-weights easy frequent-class examples to lift rare-class macro-F1.
  Validation loss stays unweighted CE.
- **Semantic embeddings** (`CONFIG["semantic_embeddings"]`, `--semantic-embeddings`) —
  `preprocessing/template_embeddings.py` initialises the event embedding from template
  TEXT so similar events start close; the model gains a `pretrained_embeddings` kwarg
  (default None → identical state_dict, so `best_model.pt` still loads).
- **Dataset registry** (`CONFIG["datasets"]`, `active_dataset()`, `config_for_dataset()`,
  `--dataset`) — makes paths dataset-agnostic. **`hdfs` and `bgl` are both wired** and validated
  (`main.py --dataset bgl --detect`, `outputs/bgl_results.md`); `openstack` is a placeholder (no
  downloads). `config_for_dataset()` overlays each dataset's cache/checkpoint/output paths on the
  HDFS-shaped defaults. Non-default training configs route to `.checkpoints/<tag>/`.

## Extension points (future work)

- **Uncertain-case routing — DELIVERED**: the former "advanced model" stub is
  superseded in PRODUCTION by the detection path (`main.py --detect`), where the
  Uncertain band is adjudicated by the constrained, scrubbed Llama classifier
  (`llm/uncertain_classifier.py`). Its measured quality lives in
  `outputs/llama_bgl_evaluation.md`; run it live with `main.py --demo-uncertain`
  or `demo_presentation.py`. (The `--hybrid` MSP path is the legacy predecessor.)
- **More models**: add e.g. `models/lstm_model.py` beside `gru_model.py`; a thin
  `models/base.py` + registry would let `main.py` select by config.
- **More LLM backends**: add `llm/openai_client.py` / `llm/anthropic_client.py`
  beside `ollama_client.py`, behind a shared `explain()` signature.
- **Serving / UI**: an `api/` (FastAPI) or `dashboard/` package can import
  `pipeline/` directly without touching the core.

# Normal-Only vs Mixed Training — Block-Level Anomaly Detection

_Task 2 (dataset: bgl). Both models are FRESH GRUs trained on the SAME honest BlockId split (stratified, seed 42); identical architecture/loss/seed, the ONLY difference is the training data. Scored on the SAME held-out TEST blocks (never in-sample). The F1 threshold is selected on VALIDATION and applied to TEST._

- Blocks: train **32,993** · val **4,714** · test **9,427**  (test anomaly base rate **0.102**)
- Normal-only training blocks: **29,616** (mixed uses all 32,993)

## Headline (best aggregator by validation PR-AUC)

| Regime | Aggregator | Precision | Recall | F1 | AUROC | PR-AUC |
|---|---|---|---|---|---|---|
| mixed | max_surprisal | 0.3738 | 0.2487 | **0.2987** | 0.5347 | **0.2230** |
| normal_only | topk_miss_frac | 0.9525 | 0.8725 | **0.9108** | 0.9932 | **0.9595** |

**Delta (normal_only − mixed):** F1 +0.6121 · AUROC +0.4585 · PR-AUC +0.7365

## All aggregators (TEST; threshold chosen on VAL)

| Regime | Aggregator | Precision | Recall | F1 | AUROC | PR-AUC | oracle-F1 |
|---|---|---|---|---|---|---|---|
| mixed | max_surprisal | 0.3738 | 0.2487 | 0.2987 | 0.5347 | 0.2230 | 0.2991 |
| mixed | mean_surprisal | 0.1508 | 0.5472 | 0.2364 | 0.4931 | 0.1260 | 0.2364 |
| mixed | topk_miss_frac | 0.1978 | 0.2456 | 0.2191 | 0.5721 | 0.1736 | 0.2340 |
| normal_only | max_surprisal | 0.7932 | 0.9979 | 0.8839 | 0.9898 | 0.9079 | 0.8839 |
| normal_only | mean_surprisal | 0.9654 | 0.7513 | 0.8450 | 0.9710 | 0.8932 | 0.8486 |
| normal_only | topk_miss_frac | 0.9525 | 0.8725 | 0.9108 | 0.9932 | 0.9595 | 0.9139 |

_oracle-F1 = best-F1 threshold picked on TEST itself (upper bound; shown only for reference — the headline F1 uses the VAL-selected threshold)._

## Measured cost (this run, wall-clock)

| Regime | Training (s) | Epochs run | Scoring val+test (s) | Blocks/s | Peak RSS (MiB) |
|---|---|---|---|---|---|
| mixed | 1343.7 | 5 | 51.9 | 272 | 1892 |
| normal_only | 1292.9 | 6 | 32.2 | 440 | 2584 |

_Device: mps. Peak RSS is process-wide and cumulative (a later regime includes memory high-water marks of earlier ones). Scoring throughput is device-synchronized over the full val+test scoring pass._

## Interpretation

DeepLog-style detection assumes the model sees NORMAL behaviour only, so anomalous transitions remain low-probability (surprising). Training on mixed data teaches the model that anomalous transitions are ordinary, so it is not surprised by them — the diagnosed root cause. A positive normal-only delta above is direct evidence for that diagnosis.


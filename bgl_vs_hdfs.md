# BGL vs HDFS — Same Pipeline, Same Protocol, Side by Side

_Both columns are measured by `run_bgl_evaluation.py` in the same run, through the identical block-level protocol: production normal-only detector checkpoint, stratified random split (seed 42), `nll_-logp_mean` surprise, validation-derived bands. Nothing is quoted from older reports unless labelled._

## Dataset shape

| | BGL | HDFS |
|---|---|---|
| Event templates (vocab) | 1,822 | 29 |
| Blocks (total) | 47,134 | 575,061 |
| Block definition | fixed 100-line window | one HDFS operation (block id) |
| Test blocks | 9,427 | 115,013 |
| Test anomaly base rate | 10.2% | 2.9% |

## Results (held-out test, val-derived thresholds)

| Metric | BGL | HDFS |
|---|---|---|
| Next-event Top-1 | 86.4% | 91.0% |
| Next-event Top-3 | 91.7% | 99.2% |
| Next-event Top-5 | 92.2% | 99.9% |
| Next-event Weighted F1 | 0.8598 | 0.9066 |
| Next-event Macro F1 | 0.1625 | 0.4120 |
| **Detection F1** | **0.9171** | **0.6740** |
| Detection Precision | 0.9822 | 0.8018 |
| Detection Recall | 0.8601 | 0.5814 |
| PR-AUC | 0.9497 | 0.6917 |
| AUROC | 0.9863 | 0.8917 |
| KS separation | 0.895 | 0.775 |

_Next-event rows use BLOCK-RESPECTING windows for both datasets (identical protocol, so BGL vs HDFS here is apples-to-apples). This differs from `hdfs_baseline_report.md`'s HDFS next-event numbers (Top-1 82.6% for the same normal-only checkpoint), which use the FLAT/global sliding-window protocol instead — the two are different measurement protocols on the same model, not a discrepancy._

### Confusion matrices (val-derived Suspicious threshold)

**BGL**

|  | Predicted Normal | Predicted Suspicious |
|---|---|---|
| **Actually Normal** | TN = 8,447 | FP = 15 |
| **Actually Anomalous** | FN = 135 | TP = 830 |

**HDFS**

|  | Predicted Normal | Predicted Suspicious |
|---|---|---|
| **Actually Normal** | TN = 111,161 | FP = 484 |
| **Actually Anomalous** | FN = 1,410 | TP = 1,958 |

## Routing distribution

| Band | BGL share (purity) | HDFS share (purity) |
|---|---|---|
| Normal | 89.4% (1.1%) | 96.7% (1.0%) |
| Uncertain | 1.7% (28.7%) | 1.1% (25.3%) |
| Suspicious | 9.0% (98.2%) | 2.1% (80.2%) |
| **LLM share** | **10.6%** | **3.3%** |

## Measured cost

| Item | BGL | HDFS | Provenance |
|---|---|---|---|
| Training (normal-only, epochs≤8, patience 3) | 1293 s (6 epochs) | 2442 s (8 epochs) | timed matched-settings `run_normal_only.py` runs (HDFS run wrote to a throwaway checkpoint dir; production HDFS checkpoints untouched) |
| Peak RSS during training | 2584 MiB | 2778 MiB | same runs |
| Detection scoring throughput | 436 blocks/s (43116 windows/s) | 4187 blocks/s (77203 windows/s) | measured this run, device-synchronized |
| Llama latency (s/call) | 1.09 | 1.19 | measured this run |
| Peak RSS (evaluation run) | 2276 MiB | (same process) | measured this run; one process evaluated both datasets |
| Device | mps | mps | — |

## Honest caveats (why BGL's higher numbers do NOT mean 'BGL is solved')

- **Base rate**: BGL anomalies are 10.2% of test blocks vs HDFS's 2.9%; F1 and PR-AUC rise mechanically with prevalence.
- **Template semantics**: BGL alerts are often intrinsically-rare hardware/kernel templates that a normal-only model has literally never seen — an easier surprise signal than HDFS's ordering anomalies among common events.
- **Synthetic sessions**: fixed 100-line windows in time order mean temporally-adjacent, near-duplicate windows can straddle a random split. The chronological-split check below quantifies exactly this optimism.
- **Chronological split**: detection F1 goes from 0.917 (random) to **0.280** (train on first 70%, test on last 20%), PR-AUC 0.950 → 0.140. The gap measures how much the random split flatters BGL; the chrono number is the deployment-realistic one.


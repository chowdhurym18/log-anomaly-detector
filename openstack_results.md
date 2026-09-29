# OpenStack Results — Suspicious-Activity Detection 

This experiment applies the same normal-only GRU anomaly detection pipeline used for HDFS and BGL to the OpenStack dataset. The GRU architecture, windowing, loss function and random seed were kept unchanged. The pipeline uses next-event prediction to calculate surprise scores, followed by Normal/Uncertain/Suspicious routing. 
## Dataset & Setup

The OpenStack dataset contains **2,067 VM-instance sessions** and **40 event templates**. Each session is the lifecycle of one VM instance, identified by its instance UUID. This is the analogue of HDFS's BlockId; BGL used fixed 100-line windows instead.

The dataset contains **4 documented anomalous VM instances**, named in the official `anomaly_labels.txt`. The other 194 instances in `openstack_abnormal.log` were treated as normal because they are not explicitly documented as anomalous. This is reversible with `openstack_preprocess.py --label-mode abnormal-file`.

The data was split by session:

- **1,444** normal training sessions
- **206** normal validation sessions
- **417** test sessions, of which **4** are the labelled anomalies (base rate **0.96%**)

All four anomalies were held out for testing. With only four labelled anomalies a stratified split would leave about one anomaly in test and none in validation, so this dataset uses `split_mode="anomaly_holdout"` (seed 42). Training is therefore normal-only by construction. Because validation contains no anomalies, the band thresholds were set from the **normal-quantile** of the validation scores (99th and 95th percentile) instead of the validation best-F1 threshold used for HDFS and BGL. HDFS and BGL keep their original protocol unchanged.

Detector: `.checkpoints/openstack/normal_only/best_model.pt` · surprise score `nll_-logp_mean` · thresholds from validation only.

## Detection Performance

| Metric | OpenStack |
|---|---:|
| **Detection F1** | **0.0000** |
| Precision | 0.0000 |
| Recall | 0.0000 |
| PR-AUC | 0.0096 |
| AUROC | 0.4177 |
| Suspicious threshold (val normal p99) | 0.2229 |
| Auto-clear threshold (val normal p95) | 0.1738 |

The test anomaly base rate is **0.0096**, which is the same as the PR-AUC. AUROC is also below 0.5, so the score ordering is not useful for separating the anomalies from normal sessions.

### Confusion Matrix

|  | Predicted Normal | Predicted Suspicious |
|---|---:|---:|
| **Actually Normal** | 403 | 10 |
| **Actually Anomalous** | 4 | 0 |

All four anomalies were missed.

### All Block-Score Definitions

I checked every block-score definition the project uses, so the result is not just an unlucky choice of aggregator.

| Block score | Precision | Recall | F1 | AUROC | PR-AUC | Distinct score values across 417 test blocks | Normal blocks tied with the anomalies |
|---|---:|---:|---:|---:|---:|---:|---:|
| max_surprisal | 0.0000 | 0.0000 | **0.0000** | 0.4177 | 0.0096 | 8 | 345 |
| mean_surprisal | 0.0000 | 0.0000 | **0.0000** | 0.4177 | 0.0096 | 10 | 345 |
| topk_miss_frac | 0.0096 | 1.0000 | **0.0190** | 0.5000 | 0.0096 | 1 | 413 |
| nll_-logp_mean ⭐ | 0.0000 | 0.0000 | **0.0000** | 0.4177 | 0.0096 | 10 | 345 |

The best AUROC across all four is **0.5000**. `topk_miss_frac` reaching recall 1.0000 is not detection: it takes a single value across all 417 test blocks, so any threshold on it flags everything. The last two columns are the important ones — the scores are heavily tied, which the sequence-identity section explains.

## Routing Distribution

| Band | Blocks | Share | Actually anomalous |
|---|---:|---:|---:|
| Normal | 379 | 90.9% | 1.1% |
| Uncertain | 28 | 6.7% | 0.0% |
| Suspicious | 10 | 2.4% | 0.0% |

The most important result here is that **all 4 anomalies were placed in the Normal band**. None reached Uncertain or Suspicious.

This is different from HDFS and BGL, where the Suspicious band held most of the anomalous blocks (80.2% and 98.2% purity respectively). The routing machinery itself works — the cutoffs put about 99% of validation-normal behaviour in the Normal band — but the underlying score carries no signal, so the Suspicious band collects ordinary sessions instead.

LLM share (Uncertain + Suspicious) is **9.1%**. The second-stage Llama classifier was **not** trained for OpenStack: with 4 positives, and 0 left for training after the holdout split, it cannot be fitted or validated. The HDFS/BGL second-stage results stand as the comparison. Run with `--no-uncertain-head`.

## Next-Event Prediction

_Same test blocks and same windows as the detector._

| Metric | OpenStack |
|---|---:|
| Top-1 accuracy | **99.3%** |
| Top-3 accuracy | 100.0% |
| Top-5 accuracy | 100.0% |
| Weighted F1 | 0.9929 |
| Weighted precision / recall | 0.9929 / 0.9929 |
| Macro F1 | 0.9930 |
| Test windows | 10,829 (16 distinct target events) |

The **99.3% Top-1 accuracy is the highest of the three datasets** (HDFS 91.0%, BGL 86.4%). This does not mean anomaly detection works well.

> ⚠️ Next-event prediction and anomaly detection are different tasks on different units.

The model is very good at predicting the next event because OpenStack sessions are highly repetitive. That contributes to the detection problem: there is very little unexpected event behaviour left for the model to be surprised by.

Macro F1 is also the highest of the three (0.9930 vs HDFS 0.4120 and BGL 0.1625). Macro F1 averages over target classes equally, and OpenStack's test windows contain only 16 distinct target events, all easy to predict. BGL's low macro F1 comes from ~1,800 mostly-rare templates. Neither number is the detection objective.

## Score Separation

| Group | Mean | Median | P90 | P99 |
|---|---:|---:|---:|---:|
| Normal | 0.034 | 0.007 | 0.130 | 0.289 |
| Anomaly | 0.007 | 0.007 | 0.007 | 0.007 |

The four anomalies all received the **same score: 0.0065**, which is why the anomaly row's mean, median, P90 and P99 are identical.

Their median is the same as the normal median (median gap **+0.000**), and their mean (0.007) is actually **lower** than the normal mean (0.034). The KS statistic is **0.165**, compared with 0.775 for HDFS and 0.895 for BGL.

The anomaly score gives no meaningful separation on OpenStack. With 4 positives the KS statistic is a weak estimate, so it is reported for consistency with the HDFS and BGL findings rather than as evidence on its own.

## Why the Score Cannot Separate the Anomalies

The reason became clear after examining the event sequences directly. This uses only the parsed logs and the labels — no model is involved.

OpenStack contains only **19 distinct event sequences across 2,067 sessions**, with one sequence accounting for **83.8%** of the traffic.

| Labelled anomaly | Events | Normal sessions with the identical sequence |
|---|---:|---:|
| `544fd51c-4edc-4780-baae-ba1d80a0acfc` | 27 | **1,729** |
| `ae651dff-c7ad-43d6-ac96-bbcd820ccca8` | 27 | **1,729** |
| `a445709b-6ad0-40ec-8860-bec60b6ca0c2` | 27 | **1,729** |
| `1643649d-2f42-4303-bfcd-7798baec19f9` | 27 | **1,729** |

All four documented anomalies have an event-template sequence that is **byte-identical to 1,729 normal sessions**.

Under the representation the detector uses, the anomalies and those normal sessions are literally the same sequence, so they receive the same score. This is visible in the 'tied with' column of the block-score table above. Changing the threshold cannot fix this — there is no information in the event sequence that separates the anomalous sessions from their normal counterparts.

This makes the result a **representation limitation rather than a GRU architecture failure**. DeepLog, LogBERT and Transformer models were **not** tested here, so this is not a measured comparison against them. The point is narrower: a sequence-only representation cannot distinguish two identical sequences, whatever model reads it.

## Recall Ceiling

To measure this directly I calculated a training-free **sequence recall ceiling**. The calculation asks: how many anomalous sessions have an exact event sequence that also occurs among normal sessions? Those sessions cannot be flagged without also flagging their normal twins, so the remaining fraction is an upper bound on recall for a sequence-only detector. No model and no training are involved.

| Dataset | Sessions | Anomalies | Anomalies sharing a normal sequence | **Recall ceiling** | Measured F1 |
|---|---:|---:|---:|---:|---:|
| HDFS *(sampled)* | 60,000 | 1,797 | 0 | **100.0%** | 0.674 |
| BGL *(sampled)* | 20,000 | 2,000 | 9 | **99.6%** | 0.917 |
| OpenStack | 2,067 | 4 | 4 | **0.0%** | 0.000 |

| Dataset | Distinct sequences / 1,000 sessions | Share of traffic in the most common sequence |
|---|---:|---:|
| HDFS | 77.2 | 16.3% |
| BGL | 273.5 | 34.5% |
| OpenStack | 9.2 | 83.8% |

_HDFS and BGL rows are computed in this run on a reproducible random subsample (60k / 20k sessions) for speed; OpenStack uses the full dataset._

The OpenStack row is the important difference. **All four anomalies collide with normal sequences, giving a 0% recall ceiling for a sequence-only detector.** That also explains why the measured OpenStack F1 is 0.000.

The recall-ceiling check can be used as a quick pre-training check on a new dataset. It does not guarantee a detector will perform well when the ceiling is high, but a ceiling of 0% means a sequence-only detector cannot separate the affected anomalies.

## Where the Signal Actually Is

The event-sequence representation does not include timestamps, so I checked session duration as a diagnostic.

| Group | N | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| Labelled anomalies | 4 | 59.9s | 61.8s | 52.9s | 74.6s |
| Normal sessions | 2,063 | 43.7s | 58.8s | 20.5s | 31879.4s |

The normal 99.9th-percentile duration is **46.9s**, while all four anomalies are between **52.9s and 74.6s**. By duration they rank **2, 3, 4, 5** out of 2,067 sessions.

Only **1** normal session is longer than the shortest anomaly, and it is a preprocessing artifact: one instance appears at the end of `normal1` and again in `normal2`, giving an 8.8-hour apparent span across a file boundary.

This suggests the OpenStack anomalies are associated with **slower execution rather than a different event order**. It is consistent with the sequence-identity result and with the dataset being built by failure injection.

The detector does not use timing information. This analysis was only used to understand the failure — **no timing-based detector was built or evaluated**. Adding a time-delta channel would change the method and make results non-comparable with HDFS and BGL.

## Measured Cost

| Item | Value | Provenance |
|---|---|---|
| Training (normal-only detector) | 20.6 s (8 epochs, early stop; best val loss 0.0308 at epoch 4) | timed `run_normal_only.py --dataset openstack` run |
| Detection scoring | 0.9 s for 623 blocks (16,171 windows) = **728 blocks/s** | measured this run, device-synchronized |
| Peak RSS (this evaluation run) | 776 MiB | measured this run |
| Device | mps | — |

## Validation

Before training, the OpenStack preprocessing passed all seven validation checks (`openstack_validate.py`, which exits non-zero on failure). Full report: `outputs/openstack/preprocessing_validation.md`.

| # | Check | Result |
|---|---|---|
| C1 | Parse coverage | **99.91%** (207,636 / 207,820 lines; the 184 unparsed lines are multi-line Python traceback continuations with no record header) |
| C2 | Our Drain parse vs LogPAI's published parse of the same 2,000 lines | **ARI = 0.990** |
| C3 | Labelled anomalies surviving preprocessing + encoding | **4/4** |
| C4 | Events dropped from anomalous sessions during encoding | **0 lost** |
| C5 | Splits disjoint; 0 anomalies in train/val; all 4 in test | PASS |
| C6 | Test event types unseen during training | **0** |
| C7 | Normal sessions from `openstack_abnormal.log` kept in test | **46** |

C2 makes a parsing or preprocessing error an unlikely explanation for the negative result: an independently built Drain parse groups the lines essentially the same way LogPAI's published parse does.

C6 matters for interpretation. No anomalous session contains an event type absent from training, so detection could never have come from spotting a new event symbol. It had to come from event order, and event order is exactly what is identical between the anomalies and normal traffic.

C7 is the provenance control. All four anomalies come only from `openstack_abnormal.log`, recorded two days before the normal files, so a detector could have scored well by learning which file a session came from. The 46 normal sessions kept from that same file test for this:

| Group | Sessions | Mean score | Median score |
|---|---:|---:|---:|
| Labelled anomalies | 4 | 0.0065 | 0.0065 |
| Normal, from `openstack_abnormal.log` | 46 | 0.0612 | 0.0065 |
| Normal, from `openstack_normal1/2.log` | 367 | 0.0308 | 0.0065 |

All three groups share the same median (**0.0065**). There is a mild provenance effect in the means (+0.0304 for abnormal-file normals), but it does not make the anomalies look suspicious: they score **below** the normal sessions from their own file and are the lowest-scoring of the three groups. This helps rule out a simple file-provenance leak.

## OpenStack vs HDFS vs BGL

_OpenStack column measured this run. HDFS and BGL columns cited from `outputs/bgl_vs_hdfs.md` and `outputs/bgl_results.md` (same block-level protocol: normal-only detector, `nll_-logp_mean` surprise, validation-derived bands, seed 42)._

| | HDFS | BGL | OpenStack |
|---|---:|---:|---:|
| Event templates | 29 | 1,822 | 40 |
| Block definition | one HDFS operation | fixed 100-line window | one VM instance lifecycle |
| Blocks (total) | 575,061 | 47,134 | 2,067 |
| Test blocks | 115,013 | 9,427 | 417 |
| Test anomaly rate | 2.9% | 10.2% | 0.96% |
| Labelled positives in test | 3,368 † | 965 † | **4** |
| Next-event Top-1 | 91.0% | 86.4% | **99.3%** |
| Next-event Macro F1 | 0.4120 | 0.1625 | 0.9930 |
| **Detection F1** | **0.6740** | **0.9171** | **0.0000** |
| Detection Precision | 0.8018 | 0.9822 | 0.0000 |
| Detection Recall | 0.5814 | 0.8601 | 0.0000 |
| PR-AUC | 0.6917 | 0.9497 | 0.0096 |
| AUROC | 0.8917 | 0.9863 | 0.4177 |
| KS separation | 0.775 | 0.895 | 0.165 |
| **Recall ceiling** | **100.0%** | **99.6%** | **0.0%** |

_ HDFS and BGL positive counts are derived from the confusion matrices published in `outputs/bgl_vs_hdfs.md` (TP + FN). Recall-ceiling rows for HDFS and BGL are computed in this run on a subsample; all other HDFS/BGL rows are cited from the committed reports._

The major difference is not simply that OpenStack has fewer event templates. Its sessions are much more repetitive, and the documented anomalies preserve the normal event ordering.

## Caveats

- **Only four documented anomalies.** One error changes recall by 25%, so the detection metrics should not be treated as statistically strong. The sequence-identity and recall-ceiling results are what carry the finding, and neither depends on the positive count.
- **The conclusion is narrow.** This experiment shows that the documented OpenStack anomalies cannot be distinguished using the event-template sequence representation used here. It does not show that sequence-based anomaly detection fails in general — HDFS and BGL in the same table are counter-examples from the same pipeline.
- **Other sequence models were not tested.** The recall-ceiling argument is based on the representation, not a direct DeepLog / LogBERT / Transformer comparison.
- **The labelling assumption matters.** Only the four instances in `anomaly_labels.txt` were treated as anomalous. If the other 194 instances in `openstack_abnormal.log` are in fact anomalous, every OpenStack number here changes.
- **Sessionization drops 73.2% of lines.** Those lines carry no instance UUID (API polling, image-cache sweeps, resource audits) and belong to no single VM. They are counted and reported, not silently dropped, but a parallel host-level session design was not tested.
- **Timing was not part of the detector.** It was only analysed to understand why the sequence-only method failed.
- **Single seed.** Results are from seed 42. Given that the anomalies are exactly tied with hundreds of normal sessions, a seed study would not change the conclusion, but it has not been run.

## Does This Support the Hypothesis?

**Hypothesis:** the existing sequence-based GRU pipeline, unchanged, generalizes from HDFS/BGL to OpenStack.

**Verdict: Not supported on this dataset — and the reason is measurable.**

The OpenStack detector achieved F1 = 0.0000, AUROC = 0.4177 and PR-AUC = 0.0096 against a base rate of 0.0096. The preprocessing passed all validation checks and the model itself was not changed.

The investigation shows that the four documented anomalies have the same event-template sequences as 1,729 normal sessions. Unlike HDFS and BGL, where the anomalies remain distinguishable from normal event sequences, the OpenStack anomalies appear to differ mainly in **execution time**.

> **The existing sequence-based approach works when the anomaly changes event behaviour, but it cannot detect an anomaly that preserves the same event sequence and only changes timing.**

The recall-ceiling check gives a simple way to identify this limitation before training a detector on another dataset.


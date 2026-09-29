# Applying the GRU Log Anomaly Detection Pipeline to OpenStack

**Research progress report — Fall 2026**
Generated from the artifacts in `outputs/openstack/`. Every number in this document
is produced by a script in the repository and can be regenerated with the commands
in §7.

---

## 1. Objective

The existing pipeline — log parsing → event templates → session sequences → GRU
next-event prediction → anomaly score → Normal/Uncertain/Suspicious routing →
selective second-stage analysis → evaluation — was built and evaluated on HDFS and
BGL. The task this term was to apply **the same pipeline, unchanged**, to a third
dataset (OpenStack) and report how well it generalizes.

The research question is therefore not *"what F1 does it get on OpenStack?"* but:

> **How well does the same sequential anomaly-detection approach generalize across
> different distributed-system log datasets — and what property of a dataset
> determines whether it will?**

**Headline answer.** The pipeline transfers from HDFS to BGL without modification
(F1 0.674 → 0.917) and **fails completely on OpenStack (F1 0.000, AUROC 0.418)**.
The failure is not a modelling deficiency: OpenStack's injected anomalies are
**invisible in the event-sequence representation**, and we can prove this
independently of any model. §4 of the comparison gives a **0% ceiling on achievable
recall** for *any* sequence-only detector on this dataset. We also identify a cheap,
training-free diagnostic that predicts this outcome in advance.

---

## 2. Dataset

The **official LogHub OpenStack dataset**, downloaded from the LogPAI Zenodo record
8196385 (`OpenStack.tar.gz`, md5 `66bd42c07837a094d9b0ea2d036b5713`, verified
against the published checksum).

| File | Lines | Role |
|---|---|---|
| `openstack_normal1.log` | 52,312 | normal traffic |
| `openstack_normal2.log` | 137,074 | normal traffic |
| `openstack_abnormal.log` | 18,434 | contains the failure-injection cases |
| `anomaly_labels.txt` | 4 UUIDs | ground truth |
| **total** | **207,820** | |

**`OpenStack_2k.log` is not used as the experiment dataset.** It was verified to be
byte-for-byte the first 2,000 lines of `openstack_normal1.log` (differing only in
CRLF line endings), contains no labelled anomalies, and covers a single 15-minute
window. It is retained only as a **parser-correctness reference**, because LogPAI
publishes their own parse of exactly those 2,000 lines (see §5, check C2).

### Ground truth

`anomaly_labels.txt` reads, in full:

> The following VM instances have injected anomalies as observed in
> openstack_abnormal.log.
> `544fd51c-…`, `ae651dff-…`, `a445709b-…`, `1643649d-…`

**Methodological decision (assumption A1).** Only these **four** VM instances are
treated as anomalous. `openstack_abnormal.log` contains 198 instances in total; the
other 194 are **not** documented as anomalous by LogPAI and are labelled Normal.
Labelling all 198 anomalous would have inflated the positive set ~50× on an
undocumented assumption. This decision is reversible in one flag:
`openstack_preprocess.py --label-mode abnormal-file`.

---

## 3. Dataset audit

Full audit: [`dataset_audit.md`](dataset_audit.md).

| Property | Value |
|---|---|
| Raw lines | 207,820 |
| Lines parsed with LogHub's published field format | 207,636 (99.91%) |
| Unparsed | 184 (0.089%) — multi-line Python traceback continuations |
| Log levels | INFO 204,506 · WARNING 3,118 · ERROR 10 · CRITICAL 2 |
| Distinct components | 12 |
| Drain templates (full dataset) | 40 |
| Time ranges | normal1 2017-05-16 00:00→06:25 · normal2 2017-05-16 15:15→2017-05-17 12:02 · **abnormal 2017-05-14 19:39→21:56** |

**Sessionization signal.** The VM instance UUID is the natural session key (the
analogue of HDFS's BlockId), but only **26.8% of lines carry one**. It appears in
two syntactic forms, and we measured both before choosing:

| | Strategy A: `[instance: <uuid>]` only | Strategy B: A **or** `/servers/<uuid>` |
|---|---|---|
| Lines kept | 51,553 (24.8%) | **55,683 (26.8%)** |
| Sessions | 2,069 | **2,069** |
| Median events/session | 25 | **27** |
| Events per labelled anomaly | 25 | **27** |

**Strategy B was adopted**: it recovers 4,130 additional events, raises every
labelled anomaly from 25 to 27 events, and creates no new sessions. `req-<uuid>` in
the ADDR field is a *request* id and is never used as a session key.

**The 73.2% of lines with no instance uuid are host-level records** (API polling,
image-cache sweeps, resource audits) that belong to no single VM and cannot be
assigned to a session. They are counted and reported, never silently dropped. This
mirrors how LogPAI's own published HDFS traces treat non-block lines.

**Leakage risk identified up front.** The 4 anomalies exist only in
`openstack_abnormal.log`, recorded **two days before** the normal files. A detector
could score well by learning *which file a session came from*. The control for this
is described in §7 and measured in §8.

---

## 4. Method

**Unchanged from HDFS/BGL.** No architectural change was made, and none was needed
— the failure mode identified here would not be fixed by one.

- **Model**: `GRUAnomalyDetector` — Embedding(64) → LayerNorm → GRU×2 (hidden 128)
  → LayerNorm → Dropout(0.3) → Linear. ~179K parameters. Identical class, identical
  hyperparameters, identical code path as HDFS/BGL.
- **Training regime**: normal-only (DeepLog-style). The GRU learns
  `P(next EventId | preceding events)` from **normal sessions only**, so anomalous
  transitions should remain low-probability.
- **Anomaly score**: per-window surprise `1 − P(actual next event)`, aggregated to
  the session. Four aggregations were evaluated (`max_surprisal`,
  `mean_surprisal`, `topk_miss_frac`, `nll_-logp_mean`).
- **Routing**: Normal / Uncertain / Suspicious by two validation-derived cutoffs.
- **Second stage**: *not adapted to OpenStack* — see §11.

> **Terminology.** Next-event prediction is the *mechanism* by which the model
> learns normal sequential behaviour. The *detector* is the anomaly score derived
> from comparing the expected next event against what actually occurred, plus the
> routing decision built on it.

### Changes made to the shared codebase

Three small, guarded, reversible changes. All existing HDFS/BGL behaviour is
byte-identical — verified by the 24 existing regression tests passing and by
`config_for_dataset("hdfs"/"bgl")` being unchanged.

| Change | File | Why |
|---|---|---|
| `split_mode="anomaly_holdout"` | `preprocessing/block_dataset.py` | With 4 positives a label-stratified split leaves ~1 anomaly in test and 0 in validation. Defaults to the original `"stratified"`. |
| `detection_band_mode="normal_quantile"` | `pipeline/detection_pipeline.py` | The anomaly-holdout split leaves validation with no positives, so the label-supervised best-F1 threshold is undefined. Defaults to the original `"supervised"`. |
| Zero-positive-validation guards | `run_normal_only.py` | Threshold and aggregator selection both divide by a positive count. Active only when validation has no positives. |

---

## 5. OpenStack preprocessing

Implemented in [`openstack_preprocess.py`](../../openstack_preprocess.py), modelled
directly on the existing `bgl_preprocess.py` and emitting the same four files the
dataset registry expects.

- **Field parsing** with LogHub's published OpenStack format:
  `<Logrecord> <Date> <Time> <Pid> <Level> <Component> [<ADDR>] <Content>`.
- **CRLF-safe**: every line is `rstrip("\r\n")`-ed before parsing.
- **Drain3 template mining** with masking for UUIDs, IPv4, long hex digests, file
  and REST paths, floats and integers. Masking is load-bearing: without it Drain
  mints a new template per VM and the vocabulary explodes from tens to thousands,
  destroying the sequential signal.
- **Outputs**: `OpenStack_structured.csv`, `OpenStack_templates.csv`,
  `Event_traces.csv`, `anomaly_label.csv`, plus `session_meta.csv` (provenance and
  timestamps — used only for evaluation controls, never by the detector).

### Validation gate

Before any training, [`openstack_validate.py`](../../openstack_validate.py) runs
seven hard checks and exits non-zero on failure. Full report:
[`preprocessing_validation.md`](preprocessing_validation.md). **All 7 passed.**

| # | Check | Result |
|---|---|---|
| C1 | Parse coverage ≥ 99.9% | ✅ 207,636/207,820 (99.91%) |
| C2 | Drain parse agrees with LogPAI's published parse | ✅ **Adjusted Rand Index 0.990** over the 2,000 reference lines |
| C3 | All 4 labelled anomalies survive preprocessing | ✅ 4/4 |
| C4 | No events dropped from anomalous sessions | ✅ 0 lost |
| C5 | Splits disjoint; 0 anomalies in train/val; all 4 in test | ✅ |
| C6 | Test-set vocabulary coverage measured | ✅ **0 unseen event types** |
| C7 | Test contains ≥20 normal sessions from the abnormal file | ✅ 46 |

C2 is the key correctness check: our independent parse induces essentially the same
clustering of log lines as LogPAI's published parse (ARI 0.990), so the results
below are not an artifact of a bad parser.

C6 matters for interpretation: **no anomalous session contains an event type absent
from training.** Detection cannot be achieved by spotting a novel symbol; it must
come from event *order*.

---

## 6. Sequence construction

- **Session key**: VM instance UUID (Strategy B, §3).
- **Ordering**: events within a session are sorted by timestamp, so the single
  instance appearing in two files stays chronological.
- **Windows**: built per session by the existing `block_dataset.build_windows` —
  context = up to 20 preceding events, target = the next event. Windows never cross
  session boundaries.
- **Result**: **2,067 sessions**, 40-template vocabulary (17 appear inside
  sessions), median 27 events per session, **4 labelled anomalous (0.194%)**.
  2 sessions with <2 events were dropped (reported, not silent).

---

## 7. Experimental setup

Fully reproducible; fixed seed 42 throughout.

| Setting | Value |
|---|---|
| Model | GRU, 2 layers, embedding 64, hidden 128, dropout 0.3 (~179K params) |
| Training data | 1,444 sessions, **normal only, 0 anomalies** |
| Validation | 206 sessions, **0 anomalies** |
| Test | 417 sessions, **4 anomalies** (0.96%) |
| Loss | sqrt-inverse class-weighted cross-entropy |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4, ReduceLROnPlateau |
| Batch size | 256 |
| Epochs | 20 max; **early stopped at 8**, best val loss 0.0308 at epoch 4 |
| Device | Apple MPS |
| Training time | 20.6 s |

**Split design and why it differs.** HDFS/BGL use a label-stratified split; with 4
positives that is not viable. The `anomaly_holdout` split puts **all 4 anomalies in
test** and splits normals 70/10/20. Consequences, all deliberate:

- Training is normal-only **by construction**, not by a later filter — there is no
  code path by which anomaly information can reach the detector.
- Validation has no anomalies, so thresholds are calibrated **unsupervised**, at the
  99th percentile of the normal validation score distribution. No label is read when
  placing the cutoffs.
- All 4 positives are evaluated rather than a fraction of 4.

**Provenance control.** Because all 4 anomalies come from the abnormal file, the test
split deliberately retains **46 normal sessions from that same file**. If those score
like the anomalies, the detector is reading file provenance; if they score like other
normals, it is reading behaviour. This is the study's main internal-validity check.

**Aggregator pre-registration.** With no validation positives, the block-score
aggregator cannot be selected on validation. `max_surprisal` was **pre-registered**
(the project's documented default in `evaluation/anomaly_eval.detection_metrics`)
rather than chosen after seeing test results. All four are reported regardless.

```bash
venv/bin/python openstack_audit.py                                    # Phase 2
venv/bin/python openstack_preprocess.py                               # Phase 3
venv/bin/python openstack_validate.py                                 # Phase 4 (gate)
venv/bin/python run_normal_only.py --dataset openstack --regimes normal_only
venv/bin/python main.py --dataset openstack --detect --no-uncertain-head
venv/bin/python openstack_evaluate.py                                 # Phase 6
venv/bin/python openstack_compare.py                                  # Phase 7
```

---

## 8. Results

Full report: [`evaluation_report.md`](evaluation_report.md).

> ⚠️ **The evaluation has 4 positives.** Recall moves in steps of 0.25 and one
> session swings F1 substantially. Precision/recall/F1 are reported for
> completeness; the rank- and separation-based findings carry the information. **No
> claim of statistical significance is made from 4 positives.**

### Detection performance

| Block score | TP | FP | FN | Precision | Recall | F1 | AUROC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| `max_surprisal` *(pre-registered)* | 0 | 38 | 4 | 0.000 | 0.000 | **0.000** | 0.418 | 0.010 |
| `mean_surprisal` | 0 | 8 | 4 | 0.000 | 0.000 | 0.000 | 0.418 | 0.010 |
| `topk_miss_frac` | 4 | 413 | 0 | 0.010 | 1.000 | 0.019 | 0.500 | 0.010 |
| `nll_-logp_mean` | 0 | 10 | 4 | 0.000 | 0.000 | 0.000 | 0.418 | 0.010 |

Base rate 0.0096. Best AUROC across **all four** score definitions: **0.500**. The
`topk_miss_frac` row reaching recall 1.000 is not detection — it flags **every**
session, because that score takes a single value across the whole test set.

**The detector carries no information about which sessions are anomalous.**

### The anomalies are not ranked — they are tied

Reporting a rank would misrepresent the result. All four anomalies receive the
**identical score**, tied with hundreds of normal sessions:

| Score | Anomaly score | Tied with | of which NORMAL | Distinct values across 417 test sessions |
|---|---|---|---|---|
| `max_surprisal` | 0.077427 | 348 sessions | **345** | **8** |
| `nll_-logp_mean` | 0.006516 | 348 sessions | **345** | **10** |
| `topk_miss_frac` | 0.000000 | 416 sessions | **413** | **1** |

This is stronger than "the anomalies were ranked poorly". They are **not ranked at
all**: the detector assigns them and a large block of normal traffic the same
number, so **no threshold placed anywhere on this score can separate them.**

### Provenance control — the negative result is genuine

| Group | Sessions | Mean `max_surprisal` | **Median** |
|---|---|---|---|
| Labelled anomalies | 4 | 0.0774 | **0.0774** |
| Normal, from `openstack_abnormal.log` | 46 | 0.3114 | **0.0774** |
| Normal, from `openstack_normal1/2.log` | 367 | 0.2132 | **0.0774** |

**All three groups share an identical median score of 0.0774** — the bulk of every
group sits at exactly the same value; the means differ only through a tail of
higher-scoring sessions.

There *is* a mild provenance effect: normal sessions from the abnormal file average
0.098 higher than those from the normal files. **But it does not manufacture
detection.** The labelled anomalies score 0.234 *below* the normal sessions of their
own file and are the **lowest-scoring of the three groups**. Whatever provenance
signal exists elevates the *non-anomalous* sessions of the abnormal file, not the
anomalies. The failure is genuine, not a leak being masked.

The anomalies being the *least* surprising group is itself informative, and D1
explains it: their event sequence is the single most common sequence in the dataset,
so the model predicts it almost perfectly. This is why AUROC sits **below** chance
(0.418) rather than at it — the ordering is weakly *inverted*, not merely
uninformative.

### Why it fails — two diagnostics

These explain the result. **Neither is part of the detector and no new detector is
proposed here.**

**D1 — The anomalous sequences are identical to normal ones.** Across all 2,067
sessions there are only **19 distinct event-template sequences**; OpenStack VM
lifecycles are highly stereotyped. Each of the four anomalies has an event sequence
**byte-identical to 1,729 sessions labelled Normal**. Under this representation the
anomalies and a large share of normal traffic are *the same object*. No sequence-only
detector — this GRU, DeepLog, LogBERT, or any other — can separate them.

**D2 — The signal exists, in a channel the representation discards.** The sequence
abstraction deliberately drops timestamps. Measuring session wall-clock duration:

| Group | n | Median | Min | Max |
|---|---|---|---|---|
| Labelled anomalies | 4 | 59.9 s | 52.9 s | 74.6 s |
| Normal sessions | 2,063 | 43.7 s | 20.5 s | 31,879 s |

Normal 99.9th percentile is **46.9 s**; all four anomalies exceed it. By duration
they rank **2nd, 3rd, 4th and 5th of 2,067 sessions**. Exactly one normal session is
longer, and it is an artifact (the cross-file instance with an 8.8-hour apparent
span).

So the injected faults are **latency anomalies**: the VM lifecycle executes in the
correct order, but slowly. The event-order channel contains no signal; the timing
channel separates the classes almost perfectly. **The pipeline reads only the first.**

---

## 9. Comparison with HDFS / BGL

Full report: [`openstack_vs_hdfs_bgl.md`](openstack_vs_hdfs_bgl.md).

| | HDFS | BGL | OpenStack |
|---|---|---|---|
| Detection F1 (routing only) | **0.674** | **0.917** | **0.000** |
| PR-AUC | 0.692 | 0.950 | 0.010 |
| AUROC | 0.892 | 0.986 | 0.418 |
| F1 with second stage | 0.745 | 0.937 | n/a (§11) |
| Event vocabulary | 29 | 1,822 | 40 |
| Session unit | one BlockId operation | 100-line window | one VM lifecycle |
| Anomaly base rate | 2.9% | 10.0% | 0.19% |
| **Distinct sequences / 1,000 sessions** | **77.2** | **273.5** | **9.2** |
| **Share of traffic in the single most common sequence** | **16.3%** | **34.5%** | **83.8%** |
| **Ceiling on achievable recall** | **100.0%** | **99.6%** | **0.0%** |

The last row is the most important number in the study. It counts anomalous sessions
whose exact event sequence *also* occurs among normal sessions — those are
unreachable for any sequence-only detector. It is computed from parsed logs alone,
**with no model and no training**, and it predicts the measured outcome on all three
datasets.

### Anomaly character

| | HDFS | BGL | OpenStack |
|---|---|---|---|
| What an anomaly *is* | a block whose operation sequence deviates | a window containing an alert-tagged fault | a VM with an **injected performance fault** |
| How it shows up | different event order | distinctive alert event types | **same events, later timestamps** |
| Visible in event order? | yes | yes | **no** |
| Visible in timing? | not needed | not needed | **yes — near-perfectly** |

### Conclusion on generalization

1. **The pipeline generalizes across datasets whose anomalies are sequential.**
   HDFS → BGL needed no model change: different system, different sessionization,
   29 vs 1,822 templates, and F1 rose from 0.674 to 0.917.
2. **It does not generalize to a different *kind* of anomaly.** OpenStack's faults
   are latency anomalies. This is a **representation mismatch**, not a modelling
   deficiency.
3. **The outcome is predictable before training.** Sequence diversity and the
   collision-based recall ceiling are computable from parsed logs in seconds. Run as
   a pre-flight check, they would have predicted this result up front. This
   diagnostic is the main contribution of this phase.

---

## 10. Failure cases

- **Complete detection failure on all four anomalies** (recall 0 at the operating
  threshold, for three of four score definitions).
- **Score collapse**: only 8–10 distinct score values across 417 test sessions;
  `topk_miss_frac` collapses to a single value. The model became so confident on the
  stereotyped lifecycle that surprise is near-zero everywhere.
- **`topk_miss_frac` degenerate operating point**: recall 1.000 at precision 0.010
  — flagging everything, which is not detection.
- **One preprocessing artifact**: instance `0f079bdd-…` appears at the end of
  `normal1` and again in `normal2`, giving an 8.8-hour apparent session spanning a
  file boundary. It is the only normal session longer than the anomalies and would
  be the single false positive of any duration-based rule. Documented, not removed.
- **Vocabulary under-use**: only 17 of 40 templates appear inside sessions; the
  other 23 are host-level events that belong to no VM and are outside the session
  abstraction entirely.

---

## 11. Limitations

1. **Four positives.** Every OpenStack detection metric is fragile. The structural
   findings (§8 D1, §9 recall ceiling) do **not** depend on the positive count and
   are the robust part of this work.
2. **Assumption A1 — the label decision.** Only the 4 documented instances are
   anomalous; the other 194 in `openstack_abnormal.log` are labelled Normal. If they
   are in fact anomalous, every OpenStack number changes. Reversible with
   `openstack_preprocess.py --label-mode abnormal-file`.
3. **Assumption A2 — sessionization.** 73.2% of lines carry no instance UUID and are
   excluded from sessions. They are host-level records not attributable to a VM, but
   an alternative design (e.g. a host-level session in parallel) was not tested.
4. **Second stage not adapted.** The Llama embedding classifier was deliberately
   **not** trained for OpenStack: with 4 positives and none available after the
   holdout split, it could not be fit or validated. Training it would have meant
   manufacturing labels. HDFS/BGL second-stage results stand as the comparison.
5. **Single seed.** Results are from seed 42. Given that the anomalies are exactly
   tied with 345 normal sessions, a seed study would not change the conclusion, but
   it has not been run.
6. **HDFS/BGL structural rows** in §9 are profiled on a reproducible random
   subsample (60k / 20k sessions) for speed; OpenStack uses the full dataset.
7. **The timing diagnostic (D2) is a measurement, not a detector.** No timing-aware
   model was built or evaluated.

---

## 12. Next steps

Ranked by evidence, not by novelty.

1. **Promote the pre-flight diagnostic into the pipeline** *(small, high value)*.
   Sequence diversity + collision-based recall ceiling, computed on any new dataset
   before training. It is already implemented in `openstack_compare.py`; extracting
   it into a reusable `dataset_feasibility.py` would let future work answer "can a
   sequence detector work here?" in seconds rather than after a full experiment.
2. **Resolve assumption A1 with the professor.** If the other 194 abnormal-file
   instances should count as anomalous, re-run with `--label-mode abnormal-file`;
   the positive set becomes 198 and all evaluation becomes statistically meaningful.
   One flag, one re-run — everything downstream is already wired.
3. **Test the timing hypothesis properly** *(a genuine method extension — flagged as
   such)*. D2 shows near-perfect separation from session duration alone. Adding a
   discretised inter-event time-delta channel to the event embedding is the standard
   DeepLog-style extension and would directly test whether the pipeline can be made
   to cover latency anomalies. **This changes the method and should not be done
   without agreement**, since it makes results non-comparable with the current
   HDFS/BGL numbers.
4. **Do not** train a second-stage classifier or change the GRU architecture for
   OpenStack. Neither addresses the identified cause, and §9 shows why.

---

## Artifact index

| File | Contents |
|---|---|
| `dataset_audit.md` | Phase 2 — full measured dataset audit |
| `preprocessing_validation.md` | Phase 4 — the 7-check validation gate |
| `preprocess_stats.json` | machine-readable preprocessing statistics |
| `normal_only_training_report.md` | Phase 5 — training run |
| `detection_report.md` | Phase 5 — Normal/Uncertain/Suspicious routing + explanations |
| `detection_results.csv` | per-session routing decisions |
| `evaluation_report.md` | Phase 6 — evaluation + diagnostics D1/D2 |
| `openstack_scores.csv` | per-session scores, ranks, provenance groups |
| `score_distribution.png` | sequence channel vs timing channel |
| `openstack_vs_hdfs_bgl.md` | Phase 7 — cross-dataset comparison |

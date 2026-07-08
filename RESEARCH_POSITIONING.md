# Research Positioning — what is novel about this project?

*Honest assessment, grounded in measured results. Every number traces to `outputs/`.
Literature F1 values are the commonly-cited figures and are **approximate** — datasets,
splits, and parsers differ, so treat them as a tier, not a ranking.*

## One-sentence positioning

A **DeepLog-class** suspicious-activity detector (a GRU learns `P(next event | history)`;
blocks with surprising transitions are flagged) wrapped in two things most log-anomaly
systems lack: a **calibrated, validation-derived Normal/Uncertain/Suspicious routing**
layer and a **hallucination-proof, evidence-grounded LLM explanation** layer — plus a
rigorous **mixed-vs-normal-only training ablation** that quantifies a failure mode the
literature usually just assumes away.

## Where we sit (measured, held-out HDFS test = 115,013 blocks)

| | Detection F1 | PR-AUC | AUROC | Notes |
|---|---|---|---|---|
| **This system — normal-only GRU** | **0.67–0.72** | **0.69** | **0.89** | `nll_mean` / `max_surprisal`, threshold chosen on validation (`outputs/detection_analysis.md`) |
| This system — *mixed* GRU (the old regime) | 0.21–0.47 | 0.13–0.43 | 0.75–0.79 | the bug we found and fixed (`outputs/normal_vs_anomaly_separation.md`) |
| DeepLog (2017) | ~0.96 | — | — | LSTM next-key; different split/setup |
| LogAnomaly (2019) | ~0.96 | — | — | + template2vec semantics + count vectors |
| LogBERT (2021) | ~0.82–0.91 | — | — | self-supervised masked-log-key transformer |
| LogGPT (2023) | reported strong | — | — | LLM as the next-key predictor |

We are honestly **below the reported SOTA on raw F1**, and §"What we do worse" says why.
But note the comparison is softer than it looks: **Le & Zhang (ICSE 2022), "Log-based
Anomaly Detection with Deep Learning: How Far Are We?"** showed those headline HDFS F1s are
substantially **inflated by random-split data leakage** (near-identical sequences in train and
test), and shrink under realistic time-based evaluation. We treat those numbers as an optimistic
tier, and — unusually — we report our *own* chronological-split number too (BGL §below), rather
than only the flattering random-split one.

### A second dataset (BGL), measured this project

The same pipeline, retrained from scratch on **BGL** (Blue Gene/L; 1,822 event templates vs
HDFS's 29), gives detection **F1 0.92 / PR-AUC 0.95 / AUROC 0.99 under a random split** — but a
**chronological split** (train on the past, test on the future) drops it to **F1 0.28**. Both
numbers are reported (`outputs/bgl_vs_hdfs.md`). The gap *is* the Le & Zhang effect, measured on
our own system; the chronological number is the honest deployment estimate.

## What we do DIFFERENTLY (the novelty)

1. **A controlled training-regime ablation with real-label metrics.** DeepLog *assumes*
   normal-only training; we *measured what happens when you don't*. Holding architecture,
   windows, loss, and seed identical and changing **only the training data**, detection on
   DeepLog's own rule (`max_surprisal`) moves **F1 0.21 → 0.72** (PR-AUC 0.13 → 0.65). The
   mechanistic proof is the top-k-miss signal going from **near-random (AUROC 0.58)** under
   mixed training to **informative (0.77)** under normal-only. This is a reproducible,
   teachable result, not a tuning trick.

2. **"Model confidence ≠ suspicion."** We show empirically that the Maximum-Softmax-
   Probability — the popular self-uncertainty signal, and the one our own earlier hybrid
   pipeline routed on — is the **worst** block-level detector (AUROC **0.44**) even though
   it is the **best** "is this prediction correct?" signal (AUC 0.87). MSP answers *"is the
   model sure of its guess?"*; detection needs *"was the actual event surprising?"* (`-log p`).
   Conflating the two is a common, rarely-stated error; we separate them with evidence
   (`outputs/detection_analysis.md`).

3. **Hallucination-proof explanations by construction.** Every verdict field
   (Expected/Observed event, Prediction-Correct, Confidence, Classification, Root-Cause,
   Recommended Investigation) is computed in **Python**; the LLM writes exactly **one**
   narrative sentence, which is then triple-scrubbed (verdict restatement → speculation →
   unsupported infrastructure cause) and, when the prediction is correct, the model is not
   even called. Result: the explanation **cannot contradict the facts** and **cannot invent
   a root cause** absent from the event templates. Most "LLM + logs" systems use the LLM as
   the detector or an unconstrained narrator; we use it as a *constrained* narrator over a
   deterministic backbone.

## What we do BETTER

- **Reproducible & streaming-scalable.** Two-pass memmap pipeline scales to 100M+ rows; one
  fixed seed; every figure regenerates from a script. LLM-as-detector approaches (LogGPT) and
  heavy transformers (LogBERT, HitAnomaly) are harder to reproduce and run.
- **Calibrated, decision-ready routing.** Normal/Uncertain/Suspicious cutoffs are *derived
  from validation* (best-F1 + a ≤1%-false-clear boundary), not guessed. On the full test set
  this auto-clears **96.7%** of blocks as Normal (only **1.0%** truly anomalous) while
  concentrating the true anomalies in the Suspicious band (**80.2%** purity).
- **Explanations that cite *why*.** Each flagged block names the surprise score and the
  validation threshold it crossed, then drills into its single most-surprising transition —
  grounded and auditable.

## What we do WORSE (stated plainly)

- **Raw detection F1 is below the reported SOTA** (~0.67–0.72 vs ~0.85–0.96). Contributing
  factors: a single global threshold (no per-event-type thresholds like DeepLog's per-key
  top-g), one training run, and HDFS_v1 split/parsing differences from the cited papers.
- **No semantic awareness.** Events are integer keys; the model cannot relate E5 "Receiving
  block" to E9 "Received block" or generalize to an unseen template. LogAnomaly/NeuralLog do.
  We tested template-text embedding init and it was **flat on HDFS** — its value is
  cross-source transfer, which we have not yet demonstrated.
- **No quantitative/count anomalies.** We detect *order* surprise, not *count* surprise
  (e.g. "too many retries"). LogAnomaly's count vectors catch that class; we do not.
- **Split-sensitive generalization.** The pipeline now runs on **two** datasets (HDFS + BGL) via
  the dataset registry, so the "dataset-agnostic" claim is tested, not just architectural. But BGL's
  strong random-split detection collapses under a chronological split (F1 0.92 → 0.28), so we do
  **not** claim robust cross-time generalization — only that the method transfers and that we
  measured its limits honestly. OpenStack/Thunderbird remain untested.
- **The LLM does not improve the decision.** On the Uncertain band, a constrained Llama 3.2 (even
  with grounded evidence, few-shot, and self-consistency) does not beat a simple keyword baseline
  (`outputs/llama_bgl_evaluation.md`); its value here is *explanation*, not classification.

## Honest bottom line (for the professor / poster)

This is not a new state-of-the-art F1. It is a **rigorously evaluated, reproducible
DeepLog-class detector** whose contributions are **methodological and engineering**: a
quantified training-regime failure mode, a clean separation of *confidence* from *suspicion*,
and a structurally hallucination-proof explanation layer. The most defensible claim is not
"our model is best" — it is **"we measured the right thing honestly, found the real
bottleneck, and fixed it without adding unjustified complexity."** The deliberate *negative*
results (BiGRU, attention, focal loss, and semantic init all flat on HDFS) are part of that
rigor, not a gap.

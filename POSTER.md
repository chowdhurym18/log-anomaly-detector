# Explainable Suspicious-Activity Detection in System Logs
### Learn normal behaviour → measure surprise → flag Normal / Uncertain / Suspicious → explain

**Bergen Community College — Research Fair**
Mohaimenul Hoque Chowdhury · mohaimenulhoquec@gmail.com

> Poster-ready copy. All numbers trace to `outputs/`. The headline detection result
> (`outputs/detection_report.md`, `outputs/normal_vs_anomaly_separation.md`,
> `outputs/detection_analysis.md`) is reproduced with `python main.py --detect`.

---

## 1. Project Summary

Modern computer systems produce millions of log lines. Hidden in them are rare
**anomalies** — the early signs of failures, data loss, or attacks — but no human
can read every line. This project builds a small neural network that **learns what
"normal" logs look like** and automatically flags sequences that break the
pattern. When something is flagged, the system writes a plain-English,
**evidence-grounded explanation** of *why*, and shows **which events the model paid
attention to**. It runs on a laptop and scales to millions of log sequences.

## 2. Problem Statement

- **Data**: HDFS (Hadoop Distributed File System) logs — 11.2M events across
  575,061 storage blocks, reduced to 29 distinct event types (E1–E29).
- **Challenge**: anomalies are rare and there are no labels at run time, so we
  cannot simply train a "normal vs. anomaly" classifier.
- **Question**: *Can a model learn the normal order of log events well enough that
  "surprising" next events reveal anomalies — and can it explain itself?*

## 3. Methodology

**Reframe detection as prediction.** Instead of labeling anomalies, the model
learns to **predict the next event** from a window of the previous 20 events
(`P(next event | history)`). A sequence is suspicious when the event that actually
happens is one the model found *unlikely*.

**The model (three versions compared).**

```
Events → Embedding → Bidirectional GRU → Attention → Classifier → next-event probabilities
```

- **GRU** (baseline): reads the window left-to-right.
- **Bidirectional GRU**: reads the window **both directions** for richer context.
  *Safe here:* the event being predicted is *outside* the input window, so reading
  the window backward never "sees the answer" — no information leakage.
- **+ Attention**: instead of remembering only the last step, the model learns a
  **weight for every event** and focuses on the ones that matter — which also tells
  us *which events drove the decision*.

**Suspicious-activity score.** For each block we slide a window and read how *surprising*
the actual next event is: the proper surprise `-log p(actual event)`. A block's score is its
average surprise. Two cut-offs are **learned from a validation set** (not hand-picked) and
sort every block into **Normal** (low surprise — auto-cleared), **Uncertain** (the ambiguous
middle — reviewed), or **Suspicious** (high surprise — flagged). We compared **nine** candidate
scores (`outputs/detection_analysis.md`) and `-log p` separated anomalies best.

**Evidence-based explanation.** Flagged anomalies get a report where **every factual
field is written by Python** (expected vs. observed event, scores, attention focus).
A local language model (Llama 3.2) writes only one short interpretation sentence,
which is automatically **scrubbed** of any claim it cannot support from the log
templates — so it **cannot hallucinate** a cause.

**Selective, grounded escalation.** Calling a language model on every block is expensive, so
only **Suspicious** (and Uncertain) blocks are explained. Each flagged block first states
*why* it was flagged — which surprise score crossed which validation-derived threshold — then
drills into its single most-surprising transition. The diagrammed flow:
*logs → templates → normal-only GRU → surprise → Normal / Uncertain / Suspicious →
(flagged only) → Llama → explanation*.

## 4. Results

**Architecture is *not* the bottleneck (a deliberate negative result).** Next-event
prediction is already near its ceiling, and all three architectures tie — so the recurrent
model is not what limits detection (`outputs/model_comparison.md`):

| Variant | Top-1 | Top-3 | Weighted F1 | Macro F1 | Speed |
|---|---|---|---|---|---|
| **GRU (baseline)** | **92.07%** | 99.67% | 0.918 | 0.469 | 22 ms/1k (2× faster) |
| Bidirectional GRU | 92.02% | 99.61% | 0.918 | 0.458 | 42 ms/1k |
| BiGRU + Attention | 91.62% | 99.67% | 0.916 | 0.458 | 43 ms/1k |

Top-3 = 99.67%, and ~70% of the Top-1 errors are three *symmetric, order-ambiguous* event
pairs — irreducible. So we stopped optimising the architecture and fixed the **training data**.

**What the attention shows (live demo).** For a flagged sequence the model reports,
e.g.:

```
Event ID    Weight
E11         0.42
E26         0.31
E5          0.17
```

→ *"These are the events the model focused on when detecting the anomaly."*

**Does the confidence routing work?** A dedicated evaluation harness
(`evaluate_hybrid.py`, since archived; results preserved in
`outputs/hybrid_evaluation.md`) checks this directly. Prediction accuracy **falls
monotonically** across the three confidence bands (Normal ≫ Uncertain > Anomaly),
which confirms that low confidence really does mark the *hard* cases — so escalating
only the Uncertain band to the LLM is well targeted. Sampled explanations were
**100% evidence-clean** (no hallucinated causes survived scrubbing). Full numbers,
the threshold study, and the confidence histogram are in
`outputs/hybrid_evaluation.md`.

**Suspicious-activity detection (the headline result).** We measure block-level detection
against the ground-truth `anomaly_label.csv`. **The training regime matters far more than the
architecture.** Trained on **all blocks (normal + anomalous mixed)**, the detector is barely
surprised by anomalies — F1 **0.21**. Trained on **normal blocks only** — *same architecture,
windows, loss, and seed; only the data changed* — detection jumps to F1 **0.72** (PR-AUC 0.65)
on DeepLog's peak-surprise rule; the production `-log p`-mean score gives **F1 0.67 · PR-AUC
0.69 · AUROC 0.89**. The mechanism is visible in the score histograms
(`outputs/separation_histograms.png`): mixed training leaves the Normal and Anomaly
distributions overlapping; normal-only pushes the anomalies cleanly to the right. The
clearest proof is the classic DeepLog flag (top-k-miss) going from **near-random (AUROC 0.58)**
to **informative (0.77)**. See `outputs/normal_vs_anomaly_separation.md`.

On the full held-out test set (115,013 blocks) the production detector **auto-clears 96.7% as
Normal** (only 1.0% truly anomalous), routes 1.1% to **Uncertain**, and flags 2.1% as
**Suspicious** — and that Suspicious band is **80.2% truly anomalous**. Thresholds are derived
from validation, not guessed (`outputs/detection_report.md`).

**Which surprise signal — and a caution about "confidence."** A 9-way scoring study
(`outputs/detection_analysis.md`) shows the proper surprise `-log p(actual event)` is the best
detector, and that the model's own **confidence** (max-softmax probability) — a popular
uncertainty signal — is the **worst** block detector (AUROC 0.44). Confidence answers *"is the
guess right?"*; suspicious-activity detection needs *"was the actual event surprising?"* We
separate the two with evidence.

## 5. Future Work — ranked by evidence (`docs/ROADMAP.md`)

**Already done here:** ground-truth block-level detection, block-boundary-aware windows
(per-block scoring, no cross-block leakage), and the **normal-only training regime** — the
headline win. What the evidence says to do next, in order:

1. **Per-event-type thresholds** (DeepLog top-g): the cheapest remaining detection-F1 gain —
   a separate sensitivity per event type instead of one global cut-off.
2. **A second dataset (BGL / OpenStack)**: train on HDFS, flag suspicious activity on a
   *different* system — the dataset registry is already wired (`--dataset`).
3. **Semantic embeddings**: *measured flat on HDFS*; worth it only paired with #2 (cross-source).

Explicitly **not** pursued — the evidence does not justify the added complexity: transformers/
attention as a detector (all tied), RAG/vector DB, agents, focal loss (macro-F1 −0.08), or a
larger LLM (the deterministic backbone already carries the explanations). See `docs/ROADMAP.md`.

---

### How to reproduce (for judges)
```bash
# The headline suspicious-activity detector:
python run_normal_only.py            # train the normal-only detector (block level)
python main.py --detect              # Normal/Uncertain/Suspicious → outputs/detection_report.md
python analyze_detection.py          # why normal-only wins → outputs/normal_vs_anomaly_separation.md
python run_scoring_comparison.py     # best surprise signal → outputs/detection_analysis.md
# Supporting / next-event story:
python compare_models.py --reuse-existing   # architecture comparison table (all tied)
python demo.py                              # live single-sequence demo (attention + explanation)
```

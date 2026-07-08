# Roadmap — future work, ranked by evidence

*Re-ranked after the Wave-1 results. Where the original audit (`docs/RESEARCH_AUDIT.md`)
guessed, we now have measurements — and they change the priority order (notably:
semantic embeddings are **measured flat on HDFS**, so they drop below a second dataset).
Each item is rated on expected **gain**, **complexity**, **research value**, and
**presentation value** (Low / Med / High).*

## Ranking at a glance

| # | Item | Expected gain | Complexity | Research | Presentation | Status |
|---|---|---|---|---|---|---|
| 0 | **Normal-only training** | **Huge** (F1 0.21 → 0.72) | Low | High | **Very High** | ✅ **DONE** |
| 2 | **Second dataset (BGL)** | High (a *generalization* result) | Med | **High** | **High** | ✅ **DONE** (with chrono caveat) |
| 1 | **Per-event-type thresholds** (DeepLog top-g) | ~0 on BGL (measured) | Low | Med | Med | ✗ **tried, failed** |
| 3 | **Semantic embeddings** | untested cross-source | Low–Med | Med | Med | built; now the recommended next |
| 4 | **Self-supervised (LogBERT-style)** | Possibly higher ceiling | High | High | Med–High | *describe, don't build* |
| 5 | **Larger explanation model** | ✗ worse calibration (measured) | Low | Low | Low–Med | tested, not adopted |

## The detail

### 0. Normal-only training — ✅ done, and it was the whole game
Verified at full scale (`outputs/normal_vs_anomaly_separation.md`): training on normal
blocks only lifts DeepLog-rule detection **F1 0.21 → 0.72**, PR-AUC 0.13 → 0.65, and turns
the top-k-miss signal from near-random (0.58) into informative (0.77). Nothing else on this
list is close. It is now the production default (`main.py --detect`).

### 2. Second dataset (BGL) — ✅ done, the real research/fair story
Verified this project (`outputs/bgl_results.md`, `outputs/bgl_vs_hdfs.md`): the same pipeline,
retrained from scratch on BGL (1,822 templates), runs end-to-end with no architecture change.
Detection **F1 0.92 / PR-AUC 0.95 under a random split** — but a **chronological split** (train
past, test future) drops it to **F1 0.28**, the Le & Zhang (ICSE'22) data-leakage effect measured
on our own system. The honest headline is "the method transfers, and we measured where the random
split flatters it," not "BGL beats HDFS." Registry wiring is now real (`config_for_dataset`,
`--dataset bgl`), not just architectural.

### 1. Per-event-type thresholds — ✗ tried on BGL, measured flat
- **What we did:** added a per-event-calibrated surprise score (`nll_perevent_z_mean/max`) that
  z-scores each window's NLL against that target event's own reference distribution.
- **Result:** **failed** — validation PR-AUC 0.21–0.30 vs the baseline's 0.95
  (`outputs/bgl/detection_analysis.md`). Mechanism: the reference distribution for a rare event
  includes its labelled-anomalous occurrences, which raises its "expected" surprise and hides
  exactly the anomalies we want to catch. A leakage trap, not a tuning miss.
- **Verdict:** do not revisit without fixing that reference-set contamination. No longer the
  recommended next step.

### 3. Semantic embeddings — now the recommended next step
- **Why:** initialise event embeddings from template text so unseen/synonym templates land
  near known ones. **Measured flat on HDFS** (`outputs/semantic_embedding_report.md`:
  macro-F1 −0.02, detection-F1 −0.02) because HDFS's 29 templates are already well-learned.
- **Gain:** its only payoff is cross-source transfer — and a second dataset (BGL, 1,822 templates)
  now exists to test it on, so it is no longer gated. This is the highest-value untried lever.
- **Complexity:** Low–Med — already built (`--semantic-embeddings`), just never run on BGL.
- **Verdict:** the natural next experiment; would also help the (currently very low) BGL macro-F1.

### 4. Self-supervised, LogBERT-style masked-log-key — *describe, don't build*
- **Why:** mask a key, predict it from both sides; score by masked-prediction error. Label-free,
  bidirectional, potentially a higher ceiling — the professor's "SSL + bidirectional + attention"
  done properly.
- **Gain:** Possibly higher detection F1; uncertain at our scale.
- **Complexity:** High — a new transformer model + training regime. **Out of scope by explicit
  project constraint** (no transformers/attention), and the audit already showed BiGRU/attention
  do not help at this scale. **Pitch it as a principled, prototyped stretch goal**, not a built
  feature — claiming it without evidence would undercut the project's rigor.

### 5. Larger explanation model — ✗ tested (llama3.2:3b), not adopted
- **What we did:** re-ran the best Uncertain-band classifier config with **llama3.2:3b** instead
  of 1b (`outputs/llama_bgl_evaluation.md`).
- **Result:** the 3B model roughly *doubled* recall but **collapsed accuracy** (~45% on both
  datasets) — more trigger-happy, not better-calibrated. It swapped the 1B model's
  over-conservatism for over-flagging; neither beats a keyword baseline on this band.
- **Verdict:** bigger did not help here. The deterministic backbone owns every verdict; the LLM
  is the explainer, not the decider. Not a detection or correctness lever.

## What is explicitly NOT on the roadmap (and why)
Per the project constraints and the audit's null results: **no** transformers/attention as a
detector (BiGRU/attention measured flat), **no** RAG or vector DB (the lever is the surprise
signal, not retrieval), **no** agent/multi-agent framework, **no** LLM-as-detector (cost,
reproducibility, hallucination), and **no** focal loss (measured macro-F1 −0.08). Each of these
adds complexity the evidence does not justify.

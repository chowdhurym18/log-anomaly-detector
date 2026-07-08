# Research Audit — GRU + LLM Log Anomaly Detection

*Senior-ML review. Every quantitative claim is traceable to a file in `outputs/`,
`docs/`, or the code. Literature numbers are flagged as approximate — verify against
the cited papers before quoting them.*

**Evidence sources:** `outputs/model_comparison.csv` (fair 3-way variant table),
`eval_full_log.txt` (production gru on the full 2.235M-row test set; archived —
regenerable via `main.py --eval-only`),
`outputs/hybrid_evaluation.md` (confidence / calibration / routing),
`outputs/explanation_quality_report.md` (explanation survival), `models/gru_model.py`,
`config.py`, and the (then-unused) `data/HDFS_v1/preprocessed/anomaly_label.csv`.

---

## 1. Executive Summary

The system is a solid, well-engineered **DeepLog-class** detector: a GRU learns
`P(next event | window)`, the max-softmax probability (MSP) gives a confidence, and
only *uncertain* cases are escalated to a local LLM for a hallucination-resistant
explanation. It is reproducible, streaming-scalable, and the explanation layer is
genuinely strong work.

Three evidence-based findings reframe the professor's feedback:

1. **Top-1 is already near its irreducible ceiling, not "stuck."** Top-3 = 99.67%
   and Top-5 = 99.94% (`eval_full_log.txt`). The model almost always has the right
   answer in its top 3. Of the 177,304 Top-1 errors, **~70% come from just three
   *symmetric* event pairs** — E22↔E5 (56.8k), E26↔E11 (38.3k), E3↔E4 (30.1k) — i.e.
   concurrent/interchangeable HDFS operations whose exact order is non-deterministic.
   No amount of sequence modeling resolves order that the logs don't contain.

2. **Bidirectionality and attention were already tried and did not help.** The fair
   reuse comparison (`model_comparison.csv`): **gru 92.07% / bigru 92.02% /
   bigru_attention 91.62%** Top-1, with macro-F1 ≈ 0.46–0.47 for all three and gru
   being 2× faster. The recurrent architecture is *not* the bottleneck.

3. **The real gaps are measurement and imbalance, not the model.** The project has
   never computed **anomaly-detection** metrics against the block-level ground truth
   it already owns (`anomaly_label.csv`) — so it cannot be compared to any paper. And
   macro-F1 (0.469) ≪ weighted-F1 (0.918): the model is excellent on frequent events
   and **scores F1 = 0.00 on several rare classes** (E7, E27, E28, E29 …).

**Top recommendation:** before any architecture change, **measure real anomaly
detection** (precision / recall / F1 / AUROC vs `anomaly_label.csv`). It is one file
of code, makes you directly comparable to DeepLog/LogBERT/etc., and is the single most
valuable thing for both the research story and the fair. Then invest in **semantic
embeddings** (research value + generalization) over chasing Top-1.

Recommended path: **Plan A now (1–2 wks)**, then **one Plan-B step** for academic
weight. Plan C (a LogBERT-style self-supervised transformer) is the honest "frontier"
answer if time allows — but pitch it as *principled*, not trendy.

---

## 2. Research Survey

Approximate HDFS F1 from the literature (⚠ verify against the paper — datasets,
splits, and parsers differ and inflate/deflate numbers).

| Method | Architecture | Strengths | Weaknesses | Reported HDFS F1 (≈) | Complexity | Suitability |
|---|---|---|---|---|---|---|
| **Your system** | GRU next-event + MSP routing + LLM explain | Reproducible, streaming, calibrated routing, anti-hallucination explanations | No real-label metric yet; log-key only (no semantics) | n/a (proxy: 92% Top-1) | Low | — (baseline) |
| **DeepLog** (2017) | LSTM next-key, top-g detection | Simple, online, unsupervised on normal, interpretable | Ignores semantics; brittle to new/unstable templates | ~0.96 | Low | ✓✓ you already are this |
| **LogAnomaly** (2019) | LSTM + template2vec + count vectors | Semantic embeddings (synonyms/new templates); catches *quantitative* anomalies | Still LSTM; needs synonym set | ~0.96 | Low–Med | ✓✓ Plan B |
| **LogBERT** (2021) | BERT, masked-log-key + hypersphere (self-supervised) | Bidirectional, self-supervised, no labels, strong | Heavier; less interpretable; needs more compute | ~0.82–0.91 | Med | ✓ Plan C (the "frontier" ask) |
| **PLELog** (2021) | Semantic embed + clustering pseudo-labels + attention BiLSTM | Semi-supervised; attention; robust to label scarcity | Clustering step adds moving parts | ~0.95–0.98 | Med | ~ borrow the semantic+attention parts |
| **HitAnomaly** (2020) | Hierarchical transformer over templates **+ parameters** | Uses parameter values; transformer fusion | Supervised (needs labels); heavier | ~0.98 | High | ✗ supervised, heavy |
| **NeuralLog** (2021) | **Parser-free**: BERT message embeddings → transformer | No parsing errors; generalizes to unseen logs/datasets | Supervised; embedding cost; less interpretable | ~0.97–0.98 | Med–High | ✓ idea for generalization (Task 5) |
| **LogGPT** (2023) | GPT/LLM next-key (sometimes RL top-k reward) | Leverages LLM priors; few-shot | Cost; prompt sensitivity; reproducibility/control | reported strong | Med–High | ~ not as core detector |
| **Recent SSL / Transformer** | Masked / contrastive pretraining, likelihood/reconstruction scoring | Label-free, transfer | Compute; tuning; interpretability | varies | Med–High | ✓ direction, via LogBERT |
| **Recent LLM-hybrid** | LLM for parsing / explanation / RAG detection | Flexible; good explanations | RAG adds infra; detection by LLM is costly & less reproducible | varies | Med–High | explanation ✓ (you do this); RAG ✗ unjustified here |

**Takeaway:** your detector sits at the DeepLog tier. The *cheap* modern upgrade is
**semantics** (LogAnomaly/NeuralLog); the *principled* frontier upgrade is
**self-supervised masked modeling** (LogBERT). Supervised parameter-heavy models
(HitAnomaly) and LLM-as-detector (LogGPT) are poor fits for an explainable,
reproducible, undergraduate project.

---

## 3. Gap Analysis

**What you already do well**
- Two-pass streaming + memmap design scales to 100M+ rows (`preprocessing/`).
- **Calibrated, evidence-validated confidence routing** — MSP beats top-2 gap and
  entropy by AUC (0.8725 vs 0.8739 vs 0.8664, `hybrid_evaluation.md`); accuracy falls
  monotonically across bands (94.4 → 61.5 → 47.1%), proving the routing is meaningful.
- **Hallucination-resistant explanations** — Python owns every verdict; the LLM gets
  one scrubbed field; survival rose 20% → ~90% after the prompt fix
  (`explanation_quality_report.md`). This is genuinely publishable engineering.

**What is missing**
- **Real anomaly-detection metrics.** `anomaly_label.csv` (575k blocks, Normal/Anomaly)
  is never used. You report a *proxy* (next-event accuracy), not detection P/R/F1/AUROC.
- **Semantic awareness.** Events are integer IDs; the model can't relate E5 ("Receiving
  block") to E9 ("Received block") or generalize to an unseen template.
- **Cross-source evaluation.** HDFS-only; the generalization goal is untested.

**What is outdated**
- Pure log-key next-event prediction (the 2017 DeepLog formulation) **without
  semantics**. The recurrent core itself is fine — the *input representation* is dated.

**What to fix first (highest value / lowest effort)**
1. **Real-label anomaly evaluation** (Plan A) — makes you comparable to every paper.
2. **Class-imbalance handling** — macro-F1 0.469 with F1 = 0.00 on E7/E27/E28/E29.
3. **Semantic embeddings** (Plan B) — research value *and* generalization.

---

## 4. Recommended Architecture (three plans)

Keep the spine: **logs → templates → encoder → model → confidence → route →
explain.** Change the *representation* and the *evaluation*, not the skeleton.

| Plan | Scope | Accuracy impact | Dev time | Risk |
|---|---|---|---|---|
| **A — Conservative** | Real-label P/R/F1/AUROC eval vs `anomaly_label.csv`; class-imbalance (focal loss / per-class thresholds); keep `gru` (proven best). | Top-1 +0–1% (not the goal); macro-F1 ↑ meaningfully; **first real detection numbers** | 1–2 wks | Low |
| **B — Moderate** | **Semantic event embeddings** (init `nn.Embedding` from template text, à la LogAnomaly/NeuralLog) + a small **Transformer-encoder** variant beside the GRU (same next-event objective, fair compare) + a **2nd dataset** (BGL/OpenStack) for cross-source. | Detection F1 ↑ on rare/new templates; modest Top-1; big generalization gain | 2–4 wks | Med |
| **C — Advanced** | **LogBERT-style self-supervised** masked-log-key Transformer (+ optional hypersphere objective), scored by masked-prediction error, evaluated on real labels. | Potentially higher detection F1; label-free; the professor's "SSL + bidirectional + attention" done properly | 4–6 wks | Med–High |

**Recommendation:** do **A**, then the **semantic-embedding half of B** (cheapest high-
value research step). Treat **C** as a stretch goal you can *describe* confidently even
if you don't finish it.

---

## 5. Top-1 Accuracy Improvement Opportunities

**Diagnosis (from `eval_full_log.txt`):**
- Top-3 = 99.67% → the answer is nearly always in the top-3; Top-1 headroom is small.
- **~70% of all errors are 3 symmetric pairs** (E22↔E5, E26↔E11, E3↔E4) — interchangeable
  concurrent events. This is **aleatoric** (irreducible) ambiguity: the order isn't in
  the data, so no sequence model can recover it.
- Rare classes are unlearned: E7 (recall 0.036), E20 (F1 0.155), E27/E28 (F1 0.000 at
  n=207/256). These barely affect Top-1 (frequency-weighted) but crush macro-F1.

| Opportunity | Expected Top-1 gain | Difficulty | Research value | Presentation value |
|---|---|---|---|---|
| **Reframe to anomaly detection (real labels)** | n/a (different, *better* metric) | Low | High | **Very High** |
| Class-imbalance: focal loss / per-class thresholds | +0–1% (macro-F1 ↑↑) | Low | Med | High (the imbalance story) |
| Semantic embeddings (template-text init) | +0.5–1.5% (rare/ambiguous) | Med | High | High |
| Quantitative/count features (LogAnomaly idea) for the symmetric pairs | +0.5–1% | Med | Med | Med |
| Transformer encoder (same objective) | ~0% expected (BiGRU already tied) | Med | Med | Med |
| Accept Top-k @ k=3 as the routing signal | reframes "accuracy" honestly | Low | Med | Med |

**Honest verdict:** **stop optimizing Top-1 directly** — it is near its ceiling and the
remaining errors are mostly irreducible. Spend the effort on (a) the *right* metric
(detection) and (b) macro-F1 via imbalance handling and semantics.

---

## 6. Generalized Log Detection Strategy (beyond HDFS)

Goal: *learn normal → detect unusual → explain*, on any source, **without breaking the
current project**.

1. **Keep the source-agnostic objective.** Self-supervised next-event (or masked-event)
   prediction needs no labels and works on any tokenized log stream — this already
   generalizes; only the *input* is HDFS-specific.
2. **Add semantic embeddings** (Plan B). Initialize event embeddings from the **template
   text** (`HDFS.log_templates.csv` → sentence embedding or hashed bag-of-words). New
   or unseen templates then land near semantically similar ones instead of being unknown
   IDs — the NeuralLog/LogAnomaly insight, the key to cross-source transfer.
3. **Dataset registry, not a rewrite.** Add a small registry in `config.py` mapping a
   source name → (parser/path, template file). The streaming/memmap pipeline already
   reads via `CONFIG["data_path"]`; a second dataset (BGL or OpenStack — both standard,
   labeled, in the LogHub collection) plugs in behind the same interface.
4. **Demonstrate transfer.** Train on HDFS, evaluate zero-/few-shot on the 2nd source;
   report detection F1. *That* is a research-fair-worthy generalization result.

**Do not** add RAG or a vector DB for "generalization" — the generalization lever here
is the *embedding space*, not retrieval. Keep HDFS as the always-working reference.

---

## 7. Explanation System Review

**Is the GRU → confidence → LLM architecture still good?** **Yes.** Confidence-gated
escalation is exactly the cost-aware pattern modern hybrid systems use, and your
routing is empirically validated (Section 3). Keep it.

| Question | Evidence-based answer |
|---|---|
| Keep explanations deterministic? | **Yes.** Python-owned verdict fields + scrubbed single LLM field is why you have *zero* hallucinations. The deterministic Deviation/Confidence Analysis sections carry the value (avg ~280 words, `explanation_quality_report.md`). |
| Increase or decrease LLM involvement? | **Keep it minimal/optional.** With the new prompt, ~90% of LLM text now survives scrubbing — good — but it adds one factual sentence on top of an already-complete deterministic report. |
| Is fine-tuning worthwhile? | **Not yet.** The prompt fix already solved "too generic" (20% → ~90% survival). The LoRA dataset + skeleton were built (`fine_tune_dataset.py`, `lora_finetune.py` — since archived) if you later want richer free-text; the evidence doesn't justify training now. |
| Would a larger model help? | **Only for free-text polish**, at a latency/cost cost (1B latency already 2–15s/case). Not needed for correctness — the deterministic backbone is the product. |
| RAG? | **No.** Unjustified by evidence; the LLM is a thin narrator, not the detector. |

**One upgrade worth considering:** once real-label detection exists, have the explanation
cite *why the block was flagged* (which event/score crossed which threshold) — grounded,
deterministic, and far more convincing at a fair than a generic narrative.

---

## 8. 30-Day Roadmap

Ranked within each week by (1) performance gain, (2) academic value, (3) research value,
(4) ease, (5) demo value.

**Week 1 — Measure what matters (Plan A core).**
- Build `evaluation/anomaly_eval.py`: aggregate per-sequence scores → block level vs
  `anomaly_label.csv`; report **precision/recall/F1/AUROC + PR curve**, threshold sweep.
- Re-run the fair variant table; lock in `gru` (proven best + fastest).
- *Deliverable:* your first paper-comparable detection numbers + a PR-curve figure.

**Week 2 — Fix the imbalance + add semantics (Plan A + half of B).**
- Add an optional **focal-loss** flag in `training/train.py`; re-measure macro-F1 and
  detection F1 (target: lift E7/E20/E27/E28 off zero).
- **Semantic embedding init** from `HDFS.log_templates.csv`; measure rare-class F1.

**Week 3 — One Plan-B step (pick ONE).**
- *Either* a small **Transformer-encoder** variant (same objective, fair compare),
- *or* a **2nd dataset** (BGL/OpenStack) behind a config registry → cross-source F1.
  (Cross-source has higher research/demo value; the transformer has higher "modern" optics.)

**Week 4 — Write up + (stretch) frontier spike.**
- Refresh `docs/POSTER.md` with detection metrics + the "Top-1 is near-ceiling" story.
- Optional **LogBERT-style SSL spike** (masked-log-key) as a described/prototyped Plan C.
- *Deliverable:* poster, results tables, an honest "what worked / what didn't" slide
  (the bidir/attention null result is a *strength* — it shows rigor).

---

## 9. Concrete Code Changes Recommended (described, not applied)

All additive, flag-gated, backward compatible — they reuse existing utilities.

1. **`evaluation/anomaly_eval.py` (NEW) — the #1 item.**
   - Load `anomaly_label.csv` → `BlockId → {Normal, Anomaly}`.
   - Reuse `collect_confidence_outputs` / `composite_score` (`evaluation/evaluate.py`) to
     score sequences; aggregate to **block level** (e.g. block anomaly score = max or
     mean sequence score, or fraction of sequences in the ANOMALY band).
   - Compute precision/recall/F1, AUROC, PR-AUC (`sklearn.metrics`), sweep the threshold,
     save `outputs/anomaly_detection_report.md` + a PR-curve PNG (matplotlib, already a dep).
   - *Note:* sequences cross block boundaries today; aggregate by the block each window's
     target event belongs to (needs the BlockId alongside the sequence — a small
     preprocessing addition in `preprocessing/dataset.py` to carry block ids).

2. **Class-imbalance — `training/train.py` + `config.py`.**
   - Add `CONFIG["loss"]` ∈ {`"weighted_ce"` (current), `"focal"`}; implement focal loss
     (γ≈2) as an alternative to the sqrt-inverse class weights. Keep the unweighted
     val-loss early-stopping rule (documented in `training/train.py`).

3. **Semantic embedding init — `preprocessing/` (new helper) + `models/gru_model.py`.**
   - From `HDFS.log_templates.csv`, build a `(vocab, emb_dim)` matrix (sentence-transformer
     if available, else hashed bag-of-words of the template text), and use it to initialize
     `nn.Embedding` (optionally `freeze=True`). Gate with `CONFIG["semantic_embeddings"]`.

4. **Transformer variant — `models/` (new class) + `config.variant_flags`.**
   - A `TransformerEncoderClassifier` (2–4 layers, same Embedding→…→Linear interface and
     the same next-event objective) selectable as a 4th `--variant`, so `compare_models.py`
     evaluates it on the identical split — apples-to-apples.

5. **Dataset registry — `config.py`.**
   - `CONFIG["datasets"] = {name: {data_path, templates, label_csv}}` + a `--dataset` flag
     in `main.py`; the pipeline already reads through `CONFIG["data_path"]`, so this is a
     thin indirection enabling BGL/OpenStack without touching the core.

**Explicitly NOT recommended:** agent frameworks, multi-agent, RAG/vector DB, LLM-as-
detector, or fine-tuning right now — none are justified by the evidence and all hurt
explainability/reproducibility.

---

### Appendix — key measured numbers (this audit)

- Variant table (`model_comparison.csv`): gru **92.07%** / bigru 92.02% / bigru_attention
  91.62% Top-1; macro-F1 0.469 / 0.458 / 0.458; gru 22 ms/1k vs ~42 ms/1k.
- Full-test gru (`eval_full_log.txt`): Top-1 92.07%, Top-3 99.67%, Top-5 99.94%;
  weighted-F1 0.918, **macro-F1 0.469**.
- Error concentration: 177,304 total errors; **E22↔E5 56.8k, E26↔E11 38.3k, E3↔E4 30.1k
  = ~70%**.
- Zero-F1 classes: E7, E12, E15, E17, E19, E27, E28, E29 (rare; E27 n=207, E28 n=256).
- Routing (`hybrid_evaluation.md`): per-band acc 94.4/61.5/47.1%; MSP AUC 0.8725.
- Explanation (`explanation_quality_report.md`): scrubber survival 20% → ~90%; 0 hallucinations.

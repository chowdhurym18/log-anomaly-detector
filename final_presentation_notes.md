# Final Presentation Notes — Anticipated Questions & Beginner Answers

_The single source for the spoken narrative and Q&A. Uses the current (post-audit)
framing and the project's canonical terminology. Every number traces to a report in
`outputs/`. Supersedes the older `TALKING_POINTS.md` / `FINAL_PRESENTATION_SUMMARY.md`
(archived — they described the pre-audit "next-event accuracy" story)._

---

## The 30-second pitch

> "Big systems write millions of log lines a day, and the rare warning signs of failure are
> buried in them. We train a small model on what **normal** activity looks like, then flag the
> blocks of activity that **surprise** it. Most blocks are obviously fine and get cleared
> automatically; a small suspicious set gets a plain-English, evidence-checked explanation; and
> a genuinely ambiguous middle gets a second look from a language model. It's fast, it's
> reproducible, and the explanations can't make things up."

## The one-sentence contribution (lead with this to the professor)

> "The contribution isn't a new accuracy record — it's **finding and fixing the real bottleneck
> honestly**: we showed that *how you train* (normal-only vs mixed) matters far more than the
> architecture, lifting detection F1 from 0.21 to 0.72; we separated *model confidence* from
> *suspicion* with evidence; and we built an explanation layer that is structurally incapable of
> hallucinating a cause."

---

## The core questions (the professor's list) — short beginner answers

**Q: Why a GRU?**
A GRU is a small, fast recurrent network that reads a sequence in order and predicts the next
item. Log analysis is exactly "given the events so far, what comes next?", so it fits. It has
~179K parameters, trains in minutes on a laptop, and — crucially — its per-step probabilities
give us a natural, interpretable **surprise** signal. We don't need anything heavier to get that.

**Q: Why not a Transformer?**
We tested whether more model power helps and it didn't. A bidirectional GRU and a GRU with
attention both landed within a fraction of a point of the plain GRU on next-event accuracy
(92.1% → 92.0% → 91.6%), while being ~2× slower — so architecture was **not** the bottleneck
(`outputs/model_comparison.md`, archived). A Transformer is the natural *next* study if we ever
hit a real ceiling; adding it now would be complexity the evidence doesn't justify. (A masked
self-supervised model like LogBERT is described as a stretch goal in `docs/ROADMAP.md`, not
claimed as built.)

**Q: Why normal-only training?**
This is the heart of the project. If you train the model on **both** normal and anomalous logs,
it learns that anomalous transitions are ordinary — so it's not surprised by them, and it fails
as a detector. If you train it on **normal logs only**, anomalous transitions stay low-probability
and therefore surprising. Holding everything else identical and changing *only* the training data,
detection F1 went from **0.21 to 0.72** (`outputs/normal_vs_anomaly_separation.md`). That's the
DeepLog assumption, and we *measured* what happens when you violate it.

**Q: What is the anomaly score?**
For one prediction, it's the **surprise**: 1 minus the probability the model gave to the event
that *actually* happened. If the model was 99% sure and it happened, anomaly score ≈ 0.01 (not
surprising). If the model gave it 0.1% and it happened anyway, anomaly score ≈ 0.999 (very
surprising). It's always between 0 and 1.

**Q: What is the confidence score?**
The mirror image: the probability the model gave to the event that actually happened =
**1 − anomaly score**. High confidence = "the model expected what happened." We deliberately keep
these two as a pair so they're easy to explain. (Note: this is *not* the same as the model's
confidence in its own top *guess* — that's a different quantity called MSP, which we showed is
the *wrong* signal for detection.)

**Q: And the detection score?**
Anomaly/confidence are per-**window** (one next-event prediction). A **block** (one operation, or
one 100-line window) has many windows. The **detection score** aggregates them — specifically the
average surprise across the block (`nll_-logp_mean`) — and that single number is what we sort
blocks by into Normal / Uncertain / Suspicious.

**Q: Why does Llama only handle the uncertain logs?**
Because that's where a second opinion is worth paying for. The detector is confident about the
Normal band (auto-clear) and the Suspicious band (auto-flag + explain). The **Uncertain** band is
the genuinely ambiguous middle — small, but exactly where extra reasoning could help. Sending only
that band to the language model keeps the expensive step rare.

**Q: Why does routing reduce compute?**
On HDFS, ~96.7% of blocks are auto-cleared as Normal and never touch the LLM; only ~3% (Uncertain
+ Suspicious) do. A language-model call costs ~1 second each; a GRU block score costs a fraction
of a millisecond. So routing means we run the cheap detector on everything and the expensive model
on a tiny slice — the LLM cost drops by roughly 30× versus explaining every block, with no loss in
what actually gets flagged (`outputs/routing_efficiency_report.md`).

---

## The hard questions (pre-empt these — they're the ones a sharp reviewer asks)

**Q: You report BGL detection F1 of 0.92, then 0.28. Which is real?**
Both are real; they measure different things, and reporting both is the honest move. **0.92** comes
from a random train/test split. But BGL blocks are consecutive 100-line windows in time, so a random
split can put near-identical neighbouring windows in both train and test — the model partly
*memorizes*. **0.28** comes from a **chronological** split (train on the earlier logs, test on the
later ones), which is what real deployment looks like. The gap is a known effect (Le & Zhang,
ICSE 2022, showed the same inflation affects the famous DeepLog/LogAnomaly HDFS numbers). We
measured it on our own system and report the honest, deployment-realistic number
(`outputs/bgl_vs_hdfs.md`).

**Q: Does the language model actually classify well?**
No — and we measured it rather than assuming. On the hardest (Uncertain) band, even with grounded
evidence, few-shot examples, structured output, and self-consistency, Llama 3.2 (1B) reaches ~77%
accuracy on HDFS but still doesn't beat a simple keyword rule; on BGL it doesn't improve at all. A
bigger 3B model doubled recall but wrecked accuracy. So the honest conclusion is: **the language
model is a good explainer, not a good decider.** The GRU stays the detector
(`outputs/llama_bgl_evaluation.md`).

**Q: If you're below the state-of-the-art F1, what's the point?**
Two answers. First, "below SOTA" is softer than it sounds: those headline numbers are inflated by
the same random-split leakage we just described, and we're one of the few to report a chronological
number at all. Second, and more importantly, the contribution is **methodological**: a controlled
training-regime ablation with real labels (the 0.21→0.72 result), a clean separation of *confidence*
from *suspicion*, and a structurally hallucination-proof explanation layer. Good empirical work is
measuring the right thing honestly and fixing the real bottleneck — which is what this does.

**Q: How do you know the LLM isn't hallucinating causes?**
Because it structurally can't. **Python** computes every verdict field (expected vs actual event,
correct/incorrect, confidence, classification, root cause, recommended next steps). The language
model writes exactly **one** sentence, which is then triple-scrubbed: any restated verdict, any
speculation word, and any infrastructure cause not present verbatim in the log templates are
deleted automatically. When the prediction is correct, the model isn't even called. This is
regression-tested in `tests/test_explanation.py` (14 tests).

**Q: Is the bidirectional model cheating (seeing the answer)?**
No. Each example predicts the event *just past* a fixed window, using only the events *inside* the
window — the target is outside it, so reading the window in either direction can't see the answer.
The tell is empirical: if there were a leak, accuracy would spike toward 100%; instead all variants
sit in the low 90s, exactly as expected.

---

## Numbers to have memorized

| Fact | Value | Source |
|---|---|---|
| Normal-only vs mixed detection F1 | **0.21 → 0.72** | `normal_vs_anomaly_separation.md` |
| HDFS production detection | F1 0.674 · PR-AUC 0.692 · AUROC 0.892 | `detection_report.md` |
| HDFS routing (Normal / Uncertain / Suspicious) | 96.7% / 1.1% / 2.1% | `detection_report.md` |
| BGL detection (random split) | F1 0.917 · PR-AUC 0.950 · AUROC 0.986 | `bgl_results.md` |
| BGL detection (chronological split) | **F1 0.28** | `bgl_vs_hdfs.md` |
| BGL scale | 4.7M lines → 1,822 templates → 47K blocks | `bgl_results.md` |
| MSP as a detector | AUROC 0.44 (worst signal) | `detection_analysis.md` |
| Llama on Uncertain (best, HDFS) | 77% acc, still < keyword rule | `llama_bgl_evaluation.md` |
| Measured cost | ~1.1 s/LLM call · 436–4,187 blocks/s scoring | `bgl_vs_hdfs.md` |

## The one-line close

> "The headline isn't a record accuracy — it's a rigorously evaluated, reproducible detector where
> we found the real bottleneck, fixed it, measured the limits honestly (including the ones that
> aren't flattering), and built explanations that can't lie."

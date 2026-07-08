# Final Demo — Exact Commands for the Live Presentation

_Copy-paste, in order. Every command runs from the repo root with the project venv.
Nothing here trains or changes results — all of it reuses the saved production
checkpoints. Total live time ≈ 3–4 minutes. Generated for the final presentation._

**Before you start (once, off-screen):** make sure Ollama is up and the model is pulled —
`ollama serve` (in another terminal) and `ollama pull llama3.2:1b`. If Ollama is down the
whole demo still works; the LLM narrative just falls back to a deterministic line.

---

## 0. Prove it's reproducible (10 seconds)

```bash
venv/bin/python tests/test_explanation.py   # 14 pass — the anti-hallucination guarantees
venv/bin/python tests/test_models.py        # 10 pass — model + attention shapes, checkpoint loads
```

**Say:** "The explanation layer is regression-tested — 14 tests lock the guarantee that the
LLM can never invent a cause or contradict the facts. These pass on every run."

---

## 1. The headline: the live end-to-end demo (≈ 60–90 s)

```bash
venv/bin/python demo_presentation.py                 # HDFS (the primary dataset)
```

This walks ONE real block from EACH band, end to end, on the exact production path:

- **✅ Normal** — high confidence, low anomaly score → auto-cleared, no LLM.
- **⚠ Suspicious** — low confidence, high anomaly score → flagged + a grounded,
  Python-owned evidence report (the LLM writes only one scrubbed sentence).
- **❓ Uncertain** — the ambiguous middle → escalated to Llama for a final NORMAL/SUSPICIOUS
  verdict, scrubbed of any unsupported cause.

**Say, pointing at the three numbers on the Normal block:** "Confidence score is the
probability the GRU gave the event that *actually* happened. Anomaly score is one minus that —
how *surprised* the model was. The detection score is the block-level average surprise; the
bands come from thresholds learned on a validation set, not hand-picked."

_Faster live run if the room is impatient:_ `venv/bin/python demo_presentation.py --max-blocks 30000`
(≈ 20 s; illustrative thresholds on a subsample). _No Ollama?_ add `--no-llm`.

---

## 2. The full report it's based on (show, don't necessarily run live)

```bash
venv/bin/python main.py --detect                     # writes outputs/detection_report.md
```

Open `outputs/detection_report.md`. **Say:** "On the full 115,013-block held-out HDFS test set:
detection F1 0.674, PR-AUC 0.692, AUROC 0.892. 96.7% of blocks auto-clear as Normal; the
Suspicious band is 80% truly anomalous. The report explains the top flagged blocks."

_(This takes ~1 minute; safe to run, but you can also just open the pre-generated report.)_

---

## 3. Generalization to a second dataset (≈ 45 s)

```bash
venv/bin/python demo_presentation.py --dataset bgl   # same pipeline, BGL supercomputer logs
```

**Say:** "Same code, same architecture, a completely different system — Blue Gene/L
supercomputer logs, 1,822 event types vs HDFS's 29. The pipeline transfers with no changes."

Then be honest (this is the strongest research point): **"Under a random train/test split BGL
detection F1 looks like 0.92 — but we also evaluated it the deployment-realistic way, training
on the past and testing on the future (a chronological split), and F1 drops to 0.28. We report
both. See `outputs/bgl_vs_hdfs.md`."**

---

## 4. If asked "does the LLM actually decide well?" (optional, show the report)

Open `outputs/llama_bgl_evaluation.md`. **Say:** "We measured it honestly against baselines.
On the hardest band, prompt engineering lifted the 1B model on HDFS from 67% to 77% accuracy,
but it still doesn't beat a simple keyword rule — so we keep Llama as the *explainer*, and the
GRU stays the *decider*. That's an evidence-based negative result, not a guess."

---

## Command cheat-sheet

| Command | What it shows | Time | Needs Ollama? |
|---|---|---|---|
| `tests/test_explanation.py` / `test_models.py` | reproducibility, guarantees | 10 s | no |
| `demo_presentation.py` | full flow, all 3 bands (HDFS) | 60–90 s | optional |
| `demo_presentation.py --dataset bgl` | generalization | 45 s | optional |
| `main.py --detect` | full HDFS report + metrics | ~60 s | optional |
| `main.py --dataset bgl --detect` | full BGL report | ~45 s | optional |
| `main.py --demo-uncertain` | just the GRU→Llama cascade | ~45 s | yes |

**Fallbacks:** every command degrades gracefully if Ollama is off (deterministic reports only).
If a live run feels slow, add `--max-blocks 30000`. If a projector can't show a terminal well,
open the pre-generated `outputs/detection_report.md` / `outputs/bgl_vs_hdfs.md` instead.

**Note on the demo dataset:** the primary demo is HDFS. The BGL evidence explanation phrases
its investigation steps in HDFS terms ("raw HDFS log lines", "BlockId") — a known cosmetic
limitation of the shared explanation layer (documented in `outputs/final_project_audit.md`);
it does not affect any detection number.

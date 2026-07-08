# docs/ — index

Read in this order.

| Doc | What it is |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map, data flow, and the **canonical terminology** (Confidence / Anomaly / Detection score, the three bands). Start here. |
| [RESEARCH_POSITIONING.md](RESEARCH_POSITIONING.md) | Honest positioning vs DeepLog / LogAnomaly / LogBERT / LogGPT, including the BGL chronological-split caveat. |
| [ROADMAP.md](ROADMAP.md) | Future work, ranked by evidence (what was tried, what worked, what failed). |
| [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) | The original (June) audit that motivated normal-only training — historical origin. |
| [POSTER.md](POSTER.md) | Research-fair poster copy. |

## Where the results live

The measured results are in [`../outputs/`](../outputs/), not in `docs/`:

- **Start:** [`../outputs/final_summary.md`](../outputs/final_summary.md) — the capstone.
- **Reviewing the code?** [`../outputs/final_project_audit.md`](../outputs/final_project_audit.md).
- **Detection numbers:** `detection_report.md` (HDFS), `bgl_results.md` + `bgl_vs_hdfs.md` (BGL).
- **What moved where:** [`../PROJECT_CLEANUP.md`](../PROJECT_CLEANUP.md).

Overlapping older presentation docs were consolidated into the `outputs/final_*` files above and
moved to `archive/docs/` — see PROJECT_CLEANUP.md.

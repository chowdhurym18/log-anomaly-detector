# Project Cleanup — Kept / Archived / Deleted

_Final-week cleanup for the presentation. Principle: **archive, don't delete.** Nothing
of any conceivable value was removed — superseded material was moved into `archive/`
(recoverable), and only genuine throwaway junk was hard-deleted. Result: the top level
went from 21 scripts → 15 and `docs/` from 11 files → 5, with no dangling references in
the kept, professor-facing docs._

> **Note for readers of the public GitHub repository:** `archive/`, the `.gru_cache*/`
> caches, the large dataset files, and a few local-only working files referenced below
> are **not pushed** (see `.gitignore`). Everything excluded is either superseded or
> regenerable from the documented pipeline steps.

## Kept — the files that matter for development, evaluation, reproducibility, presentation

**Entry point & production path**
| File | Why kept |
|---|---|
| `main.py` | CLI entry point (`--detect`, `--quick-test`, `--demo-uncertain`, `--dataset`) |
| `config.py` | single source of truth; `config_for_dataset()` wires HDFS + BGL |
| `pipeline/detection_pipeline.py` | the PRODUCTION suspicious-activity detector |
| `pipeline/anomaly_pipeline.py`, `pipeline/hybrid_pipeline.py` | legacy paths, kept (power `--quick-test`/`--hybrid`); banner-marked legacy |
| `pipeline/uncertain_demo.py` | the `--demo-uncertain` cascade |
| `preprocessing/`, `models/`, `training/`, `evaluation/`, `llm/`, `utils/` | the package — unchanged |
| `tests/test_explanation.py`, `tests/test_models.py` | the regression suites (must stay green) |

**Active analysis / report scripts**
| File | Produces |
|---|---|
| `demo_presentation.py` | **the live demo** — one block per band, production path (new) |
| `run_normal_only.py` | mixed-vs-normal-only training + the detector checkpoint |
| `run_scoring_comparison.py` | `outputs/[bgl/]detection_analysis.md` |
| `run_bgl_evaluation.py` | `outputs/bgl_results.md`, `bgl_vs_hdfs.md` (new this phase) |
| `evaluate_llm_ladder.py` | `outputs/llama_bgl_evaluation.md` (new this phase) |
| `evaluate_uncertain.py` | `outputs/llama_uncertain_evaluation.md` |
| `analyze_detection.py`, `analyze_routing.py`, `baseline_hdfs.py`, `compare_models.py`, `demo.py` | professor-item evidence + the attention demo |
| `hdfs_v1_preprocess.py`, `bgl_preprocess.py` | build the two datasets |

**Current reports** (`outputs/`): `detection_report.md`, `detection_results.csv` (local-only:
5 MB, regenerable), `bgl_results.md`,
`bgl_vs_hdfs.md`, `llama_bgl_evaluation.md`, `final_summary.md`, `final_project_audit.md`,
`final_demo_commands.md` + `final_presentation_notes.md` (local-only speaker notes),
`hdfs_baseline_report.md`,
`routing_efficiency_report.md`, `normal_vs_anomaly_separation.md`, `detection_analysis.md`,
`anomaly_scoring_comparison.md`, `outputs/bgl/*`.

**Canonical docs** (`docs/`): `ARCHITECTURE.md` (incl. the terminology canon), `RESEARCH_POSITIONING.md`,
`ROADMAP.md`, `RESEARCH_AUDIT.md` (historical origin), `POSTER.md`, plus `docs/README.md` (index).

## Archived — moved to `archive/`, recoverable, referenced by nothing kept

**`archive/scripts/`** (superseded or one-off; verified imported by no kept module)
| File | Reason |
|---|---|
| `run_generalization.py` | superseded by `run_bgl_evaluation.py` (proper registry wiring + chrono split) |
| `evaluate_hybrid.py` | legacy MSP-hybrid evaluation harness |
| `evaluate_explanations.py` | produced the (superseded) explanation-quality report |
| `compare_training.py` | one-off focal/semantic training comparison |
| `analyze_results.py`, `run_final_comparison.py`, `analyze_attention.py` | one-off analyses, referenced by nothing |

**`archive/docs/`** (overlapping presentation docs, consolidated forward into the new
`outputs/final_*` deliverables + the canonical docs)
| File | Folded into |
|---|---|
| `PROFESSOR_SUMMARY.md`, `FINAL_PRESENTATION_SUMMARY.md`, `TALKING_POINTS.md` | `outputs/final_presentation_notes.md` |
| `PHASE_RESULTS.md`, `MEETING_NOTES.md` | `outputs/final_summary.md` |
| `CLEANUP_AUDIT.md` | this file |

**`archive/outputs/`**: `generalization_report.md` (superseded by `bgl_results.md` + `bgl_vs_hdfs.md`).

## Retained in place but SUPERSEDED (do not read for current numbers)

These legacy reports use the pre-audit **composite anomaly score** or **MSP = confidence**
definitions and were *not* moved, because foundational audit docs still cross-reference them;
they are marked here so nobody quotes them as current:
`hybrid_evaluation.md`, `hybrid_explanations_sample.md`, `hybrid_results.csv`,
`explanation_quality_report.md`, `threshold_study.csv`. Also historical (next-event ablations,
still valid as negative-result evidence): `model_comparison.{md,csv}`, `focal_loss_comparison.md`,
`semantic_embedding_report.md`.

## Deleted — genuine throwaway only

| Path | Reason |
|---|---|
| `.DS_Store` | macOS Finder junk |
| `.checkpoints/timing_hdfs/` | this week's HDFS training-time measurement; number is captured in `bgl_vs_hdfs.md`, model itself disposable |
| `.checkpoints/bgl/chrono/` | chronological-split model; `run_bgl_evaluation.py` retrains it fresh each run, so the saved file is regenerable |

## Kept in place, documented (not deleted, not archived)

- `.gru_cache/` (895 MB) — the HDFS memmap cache; regenerable but slow to rebuild, so left in place.
- Experimental checkpoints `.checkpoints/{bigru, bigru_attention, experiment, compare/*, bgl/mixed}` —
  left so `compare_models.py --reuse-existing` and the variant story still work. Production
  checkpoints (`best_model.pt`, `normal_only/normal_only`, `bgl/normal_only`) are untouched
  (byte-identical, verified in `final_verification.md`).

# Final Verification

_Every important workflow run after the final-week polish, with the guardrail assertions.
All checks pass — the polish changed presentation and documentation only; the production
pipeline behaves identically. Reproduce any row from the repo root with the project venv._

## Guardrails (behavior must be identical)

| Check | Baseline | After polish | Result |
|---|---|---|---|
| HDFS `outputs/detection_results.csv` md5 | `364d52d859634b5e137c128f29c22532` | `364d52d859634b5e137c128f29c22532` | ✅ **identical** |
| `.checkpoints/best_model.pt` md5 | `f01a23c1626884ed32f4865d6eff45a9` | unchanged | ✅ untouched |
| `.checkpoints/normal_only/normal_only/best_model.pt` md5 | `ebf5875eeb0deced09ce5c39e96dcae5` | unchanged | ✅ untouched |
| `.checkpoints/bgl/normal_only/best_model.pt` md5 | `867dde32b15429cc00ea6b1e22c05712` | unchanged | ✅ untouched |

## Workflows

| # | Workflow | Command | Result |
|---|---|---|---|
| 1 | Syntax (all modules) | `python3 -c "import ast,glob; …"` | ✅ Syntax OK |
| 2 | Explanation tests | `venv/bin/python tests/test_explanation.py` | ✅ 14/14 pass |
| 3 | Model tests | `venv/bin/python tests/test_models.py` | ✅ 10/10 pass |
| 4 | Quick-test (HDFS) | `main.py --quick-test --quick-test-n 20` | ✅ runs (Top-1 95% on the sample) |
| 5 | **Detection (HDFS)** | `main.py --detect` | ✅ runs; results CSV checksum **unchanged** |
| 6 | **Detection (BGL)** | `main.py --dataset bgl --detect` | ✅ Bands N/U/S = 8425/157/845, **F1 0.917, PR-AUC 0.950** |
| 7 | Live demo (HDFS) | `demo_presentation.py --dataset hdfs` | ✅ all **3 bands** shown + Llama cascade |
| 8 | Live demo (BGL) | `demo_presentation.py --dataset bgl` | ✅ all **3 bands** shown |
| 9 | Kept scripts import | `import main, demo_presentation, run_bgl_evaluation, …` | ✅ all import OK |
| 10 | No dangling imports | grep for archived-script imports in kept `*.py` | ✅ none (archive is self-contained) |

## CLI surface (all functional)

`--detect`, `--dataset {hdfs,bgl}`, `--quick-test [--quick-test-n N]`, `--demo-uncertain`,
`--hybrid` (legacy), `--variant {gru,bigru,bigru_attention}`, plus the standalone
`demo_presentation.py`, `run_normal_only.py`, `run_scoring_comparison.py`, `run_bgl_evaluation.py`,
`evaluate_llm_ladder.py`, `evaluate_uncertain.py`.

## Documentation & tidiness

- **Terminology** is consistent across the kept docs (canon defined once in `docs/ARCHITECTURE.md`).
- **Reports**: 20 current `outputs/*.md` (superseded ones labelled in `PROJECT_CLEANUP.md`).
- **Repo tidiness**: root scripts 21 → **15**; `docs/` 11 → **6** (5 canonical + index);
  `archive/` holds 7 scripts + 6 docs + 1 superseded report; production checkpoints untouched.
- **Deliverables present**: `outputs/final_project_audit.md`, `outputs/final_demo_commands.md`,
  `outputs/final_presentation_notes.md`, `PROJECT_CLEANUP.md`, `final_verification.md`.

## Notes / known limitations (from the audit, unchanged by design)

- The BGL evidence explanation phrases investigation steps in HDFS terms ("raw HDFS log lines",
  "BlockId") — cosmetic; the tested scrubber code was intentionally not edited. HDFS is the
  primary demo dataset.
- The legacy composite (`main.py` default) and `--hybrid` routing paths are retained (they power
  `--quick-test`) and banner-labelled as superseded; production detection is `--detect`.

**Bottom line: nothing is broken. The production path is byte-for-byte unchanged, both datasets
detect and demo end-to-end, all tests pass, and the repo is consistent and presentation-ready.**

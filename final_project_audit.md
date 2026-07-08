# Final Project Audit

_A pre-submission review of the whole project, component by component, in the spirit of a
reviewer reading a paper before it goes out. Every code change made as a result was
**behavior-preserving** and verified against three guardrails: the HDFS `detection_results.csv`
checksum (`364d52d8…`, unchanged), both regression suites (14 + 10 tests, green), and the
production checkpoint md5s (byte-identical). See `final_verification.md` for the proof and
`PROJECT_CLEANUP.md` for the file dispositions._

## Overall verdict

The project is in good shape: a clean package layout, a genuinely reproducible pipeline (fixed
seeds, checksum-stable outputs), strong regression tests on the safety-critical explanation
layer, and — unusually — a track record of **honest negative results**. The issues found were
about **clarity and consistency**, not correctness of the production path. Most were fixed in
docs or with in-place comments; a few are documented as known limitations rather than changed,
because changing them would alter tested behavior in the final week. No production code logic,
model, or checkpoint was modified.

## Findings and dispositions

| # | Finding | Severity | Disposition |
|---|---|---|---|
| 1 | **"Confidence" and "anomaly score" were defined two different ways** — the production detector uses confidence = P(actual event) / anomaly = surprisal, while the legacy composite and MSP paths (and the reports they generated) use different formulas. A reviewer reading two reports would see the same word mean two things. | **Medium** | Fixed in docs: added the **terminology canon** to `docs/ARCHITECTURE.md`; marked the legacy reports SUPERSEDED (`PROJECT_CLEANUP.md`); added clarifying banners to the legacy pipeline files. |
| 2 | **Stale documentation claims** — `RESEARCH_POSITIONING.md` said "single dataset, BGL untested"; `ROADMAP.md` still ranked BGL and per-event thresholds as future/next; `CLAUDE.md` said `anomaly_label.csv` is "not currently used"; `README.md`/`ARCHITECTURE.md` described the legacy path and `advanced_model` as current. | **Medium** | Fixed in each doc with the measured two-dataset reality (incl. the chronological-split caveat) and the current production framing. |
| 3 | **`advanced_model()` is a vestigial stub** (`pipeline/anomaly_pipeline.py`) — a routing target for uncertain sequences on the legacy path that only logs; production sends uncertain blocks to the Llama classifier instead. Misleading to a reader. | **Low** | Docstring rewritten in place to state it is vestigial and point to the production Llama path; behavior unchanged. |
| 4 | **Three coexisting routing paths** — legacy composite (`main.py` default), `--hybrid` (MSP), and production `--detect`. Only the last is production; the other two are easy to mistake for current. | **Low–Med** (clarity) | Banner comments added to both legacy files; `README.md`/`ARCHITECTURE.md` now state `--detect` is production. Kept (not removed) — `--quick-test` rides the legacy path and removing it would change behavior. |
| 5 | **BGL evidence explanations use HDFS-phrased investigation steps** — `llm/explanation._recommended_investigation` hardcodes "raw HDFS log lines" and "BlockId", which read slightly wrong on a BGL block. | **Low** (cosmetic) | **Documented, not changed.** The function is covered by `tests/test_explanation.py`; editing it risks the tested scrubber behavior in the final week, and the primary demo dataset is HDFS (where the wording is correct). Noted in `final_demo_commands.md`. |
| 6 | **`llm/uncertain_classifier.py` v1 system prompt hardcodes "for HDFS"** — was literally wrong when `run_generalization.py` ran v1 on BGL data. | **Low** | Resolved by archiving `run_generalization.py`: v1 is now only ever invoked on HDFS (`evaluate_uncertain.py`, `--demo-uncertain`), so the label is accurate. The dataset-aware v2 classifier (added earlier) is what runs on BGL. No code change needed. |
| 7 | **Repo clutter** — 21 top-level scripts (several superseded/one-off) and 11 `docs/` files (6 overlapping presentation docs) made it hard to tell what is current. | **Low** (clutter) | Archived 7 scripts + 6 docs + 1 superseded report to `archive/` (import-/citation-checked first). Root scripts 21→15, docs 11→5. See `PROJECT_CLEANUP.md`. |
| 8 | **No linter/formatter configured** — `pyflakes`/`black` are not installed, so unused imports aren't caught automatically. | **Low** | Documented as a recommendation. Unused-import removal was deliberately **not** attempted (can't verify safely without the tool; risk > reward in the final week). |

## Component-by-component

- **Preprocessing** (`preprocessing/`) — Solid. The two-pass streaming design (vocab pass, then
  memmap windows) genuinely scales and is well-commented. `StreamingLabelEncoder` silently drops
  out-of-vocab events, which is correct for a fixed-vocab detector; `cache_is_valid()` guards the
  stale-cache failure mode. `block_dataset.py` cleanly separates the block-level path. No changes.
- **Model** (`models/gru_model.py`) — One class, three variants via flags; defaults reproduce the
  original checkpoint (verified: `test_models.py` loads it `strict=True`). No changes.
- **Training** (`training/train.py`) — Weighted CE (train) + unweighted CE (val) is a deliberate,
  documented choice to keep early-stopping honest under class imbalance. No changes.
- **Evaluation** (`evaluation/`) — `evaluate.py` (next-event) and `anomaly_eval.py` (block
  detection with 2×2 confusion, AUROC, PR-AUC) are distinct and correctly kept separate. No changes.
- **Detection pipeline** (`pipeline/detection_pipeline.py`) — The production path. Bands derived on
  validation (not hand-picked), score aggregation reused from `run_scoring_comparison`. Clean. The
  `out_dir` parameter (added earlier for BGL) keeps HDFS output paths identical. No new changes.
- **Routing layer** — Sound; the clarity issues (#1, #4) were about *documentation* of it, not the
  logic. The validation-derived Normal/Uncertain/Suspicious cutoffs are the right design.
- **Llama / explanation** (`llm/`) — The strongest part for rigor: Python owns every verdict field,
  the LLM writes one triple-scrubbed sentence, and it's regression-tested. v1/v2 split is clean
  (v1 untouched for back-compat). Only cosmetic HDFS-phrasing (#5) noted.
- **Reports** (`outputs/`) — Now consistent after the terminology pass; superseded ones labelled.
- **Docs** (`docs/`) — Consolidated 11→5 canonical files + an index (`docs/README.md`).
- **CLI** (`main.py`) — Coherent; `--detect` / `--dataset` / `--quick-test` / `--demo-uncertain`
  are the ones that matter and all work. The legacy default path is now banner-labelled.
- **Config** (`config.py`) — `config_for_dataset()` is a clean registry resolver; HDFS defaults are
  byte-identical for `dataset="hdfs"`. No changes this phase.
- **Tests** (`tests/`) — Good coverage of the safety-critical layer; both suites green throughout.

## What was deliberately NOT done (and why)

- **No production logic, model, or checkpoint changes.** The whole point of the phase was polish
  without risk; every guardrail confirms behavior is identical.
- **No removal of the legacy routing paths** (#4) — `--quick-test` depends on one, and removal
  would change `main.py`'s default behavior. They are labelled, not deleted.
- **No edit to the tested explanation scrubbers** (#5) — the risk to the 14 passing tests in the
  final week outweighs a cosmetic BGL wording fix.
- **No unused-import sweep** (#8) — not safely verifiable without a linter installed.

## Recommendations (post-presentation, optional)

1. Add `black` + `ruff`/`pyflakes` to the dev deps and run once (catches #8 cleanly).
2. Parameterize the explanation layer's investigation-step wording by dataset (fixes #5) with a
   new test, when there's time to touch tested code.
3. Consider retiring the legacy composite/hybrid paths entirely once `--quick-test` is reworked
   onto the detection path — that would collapse the three-path confusion to one.

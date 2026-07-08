# =============================================================================
# pipeline/anomaly_pipeline.py — Batched anomaly detection + routing.
#
# ⚠ LEGACY (pre-audit) PATH. This is the composite-anomaly-score routing that
# `main.py` runs by DEFAULT (no flag). The PRODUCTION suspicious-activity
# detector is `main.py --detect` (pipeline/detection_pipeline.py): a normal-only
# GRU + block-level surprise (`nll_-logp_mean`) + validation-derived bands. This
# file is kept because it still powers `--quick-test` (run_quick_test, below) and
# the sample-evaluation printout, but for detection numbers use `--detect`.
#
# Scores the test set in batches, calibrates thresholds from the training-score
# distribution, classifies each sequence (Normal / Uncertain / Anomaly), and
# routes anomalies to the LLM explanation layer and uncertain cases to the
# advanced-model stub. Also holds the fast quick-test path (fixed threshold,
# no LLM, CSV output).
#
# Terminology note: here "anomaly score" is the COMPOSITE score
# (surprisal + entropy + top-k-miss; evaluation.evaluate.composite_score) and
# "confidence" = 1 − that score. The production detector defines anomaly score as
# pure per-window surprisal (1 − P(actual)); the two are different quantities.
# =============================================================================

import logging

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional

from config import CONFIG
from utils.format import format_pct, band_label
from preprocessing.encoder import StreamingLabelEncoder, validate_encoded_values
from evaluation.evaluate import collect_anomaly_outputs, choose_thresholds
from llm.explanation import send_to_llm

log = logging.getLogger(__name__)


def advanced_model(log_sequences):
    """
    VESTIGIAL STUB from the pre-audit design — do not mistake this for the
    production uncertain-case handler. In the current architecture, UNCERTAIN
    blocks are adjudicated by the constrained, scrubbed Llama classifier
    (llm.uncertain_classifier.classify_uncertain), reachable via `--detect` and
    the `--demo-uncertain` cascade. This function only logs, and only on the
    legacy composite-score path below; it is kept so that path's behaviour is
    unchanged. The "heavier model" it once stood in for is now that LLM stage.

    Args:
        log_sequences (list[list[str]]): Batch of decoded EventId sequences.
    """
    for seq in log_sequences:
        log.info(f"[→ Advanced Model] {seq}")


def run_anomaly_pipeline(model, X_path, y_path, train_idx, test_idx, encoder, device, config,
                         event_context: Optional[Dict[str, str]] = None):
    """
    Runs batched anomaly detection over the entire test set, then prints
    a human-readable report for a random sample.

    Batched inference:
        All test sequences are scored in GPU batches. The composite
        anomaly_score for each sequence combines surprisal, output entropy,
        and a top-K miss penalty (see collect_anomaly_outputs).

        Results are collected as numpy arrays and then processed at once —
        no per-sample Python overhead.

    Args:
        model     : Trained GRUAnomalyDetector.
        X_path    : Path to memmap X array.
        y_path    : Path to memmap y array.
        train_idx : Array of train-set indices (used to calibrate thresholds).
        test_idx  : Array of test-set indices.
        encoder   : Fitted StreamingLabelEncoder.
        device    : Compute device.
        config    : CONFIG dict.
        event_context : Optional EventId → template mapping for explanations.
    """
    num_print = config["num_eval_samples"]

    log.info("Running batched anomaly pipeline ...")
    train_scores, _, _, _ = collect_anomaly_outputs(
        model, X_path, y_path, train_idx, encoder, device, config, name="train"
    )
    uncertain_threshold, anomaly_threshold = choose_thresholds(train_scores, config)

    anomaly_scores, true_labels, pred_labels, sequences = collect_anomaly_outputs(
        model, X_path, y_path, test_idx, encoder, device, config, name="test"
    )

    validate_encoded_values("anomaly pipeline sequences", sequences, encoder)
    validate_encoded_values("anomaly pipeline true labels", true_labels, encoder)
    validate_encoded_values("anomaly pipeline model predictions", pred_labels, encoder)

    log.info(
        "Train anomaly-score stats: min=%.4f mean=%.4f median=%.4f max=%.4f",
        float(train_scores.min()), float(train_scores.mean()),
        float(np.median(train_scores)), float(train_scores.max()),
    )
    log.info(
        "Test anomaly-score stats: min=%.4f mean=%.4f median=%.4f max=%.4f",
        float(anomaly_scores.min()), float(anomaly_scores.mean()),
        float(np.median(anomaly_scores)), float(anomaly_scores.max()),
    )

    # ---- Global counts ----
    is_anomaly   = anomaly_scores >= anomaly_threshold
    is_uncertain = (~is_anomaly) & (anomaly_scores >= uncertain_threshold)
    is_normal    = ~(is_anomaly | is_uncertain)

    log.info(f"Pipeline results over {len(anomaly_scores):,} test sequences:")
    log.info(f"  ✅  Normal    : {is_normal.sum():,}  ({100*is_normal.mean():.1f}%)")
    log.info(f"  ❓  Uncertain : {is_uncertain.sum():,}  ({100*is_uncertain.mean():.1f}%)")
    log.info(f"  ⚠   Anomaly   : {is_anomaly.sum():,}  ({100*is_anomaly.mean():.1f}%)")

    # ---- Batch-route to downstream functions ----
    # Cap how many anomalies get an LLM explanation (config 'max_llm_explanations';
    # None/0 = explain every anomaly). Capping the INDICES here also avoids decoding
    # cases we won't explain. The highest-scoring anomalies are explained first.
    # Detection counts/metrics logged above are unaffected — only the LLM-routed
    # subset is limited.
    anomaly_idx = np.where(is_anomaly)[0]
    max_expl = config.get("max_llm_explanations")
    if max_expl and len(anomaly_idx) > max_expl:
        top = np.argsort(anomaly_scores[anomaly_idx])[::-1][:max_expl]
        anomaly_idx = anomaly_idx[top]
        log.info("Explaining the top %d of %d anomalies via LLM (config 'max_llm_explanations').",
                 max_expl, int(is_anomaly.sum()))

    # Collect decoded sequences and model metadata for anomalies.
    anomaly_cases = [
        {
            "sequence": encoder.inverse_transform(sequences[i], context=f"anomaly routing row {i}"),
            "predicted_event": encoder.inverse_transform(
                [int(pred_labels[i])], context=f"anomaly predicted label row {i}"
            )[0],
            "actual_event": encoder.inverse_transform(
                [int(true_labels[i])], context=f"anomaly true label row {i}"
            )[0],
            "confidence":      float(1.0 - anomaly_scores[i]),
            "anomaly_score":   float(anomaly_scores[i]),
            "classification":  "ANOMALY",
        }
        for i in anomaly_idx
    ]
    uncertain_seqs = [
        encoder.inverse_transform(sequences[i], context=f"uncertain routing row {i}")
        for i in np.where(is_uncertain)[0]
    ]

    if anomaly_cases:
        send_to_llm(anomaly_cases, event_context)
    if uncertain_seqs:
        advanced_model(uncertain_seqs)

    # ---- Print a random sample of `num_print` sequences ----
    rng       = np.random.default_rng(CONFIG["random_seed"])
    sample_idx = rng.choice(len(anomaly_scores),
                            size=min(num_print, len(anomaly_scores)),
                            replace=False)

    print("\n" + "=" * 72)
    print(f"  Sample Evaluation ({min(num_print, len(anomaly_scores))} sequences)")
    print("=" * 72)

    for i in sample_idx:
        score     = anomaly_scores[i]
        conf      = 1.0 - score
        true_ev   = encoder.inverse_transform(
            [int(true_labels[i])], context=f"sample true label row {i}"
        )[0]
        pred_ev   = encoder.inverse_transform(
            [int(pred_labels[i])], context=f"sample predicted label row {i}"
        )[0]
        seq_str   = encoder.inverse_transform(
            sequences[i].tolist(), context=f"sample sequence row {i}"
        )

        if score >= anomaly_threshold:
            band, badge = "ANOMALY", "⚠ "
        elif score >= uncertain_threshold:
            band, badge = "UNCERTAIN", "❓"
        else:
            band, badge = "NORMAL", "✅"

        print(f"  Seq  : {seq_str}")
        print(f"  Pred : {pred_ev:<6}  Actual : {true_ev:<6}  "
              f"Confidence : {format_pct(conf)}  Anomaly Score : {format_pct(score)}")
        print(f"  Decision → {badge}  {band_label(band)}")
        print("-" * 72)


def run_quick_test(model, X_path: str, y_path: str, test_idx: np.ndarray,
                   encoder: StreamingLabelEncoder, device: torch.device,
                   config: dict, n_samples: int = 500,
                   output_csv: str = "outputs/quick_test_results.csv") -> None:
    """
    Scores a random subset of test_idx and saves per-sequence results to CSV.

    Uses CONFIG["anomaly_threshold"] directly (fixed mode) so no training-set
    scoring pass is needed — keeping wall-clock time under a minute for any
    reasonable n_samples.

    Args:
        model       : Loaded GRUAnomalyDetector in eval state.
        X_path      : Path to memmap X array.
        y_path      : Path to memmap y array.
        test_idx    : Full test-set index array (produced by build_dataloaders).
        encoder     : Fitted StreamingLabelEncoder.
        device      : Compute device.
        config      : CONFIG dict.
        n_samples   : Number of test sequences to sample (capped at len(test_idx)).
        output_csv  : Destination file for per-sequence results.
    """
    n = min(n_samples, len(test_idx))
    rng    = np.random.default_rng(config["random_seed"])
    subset = rng.choice(test_idx, size=n, replace=False)

    log.info("Quick-test: scoring %s randomly sampled test sequences ...", f"{n:,}")

    scores, true_labels, pred_labels, sequences = collect_anomaly_outputs(
        model, X_path, y_path, subset, encoder, device, config, name="quick_test"
    )

    # Fixed thresholds — avoids the full training-set scoring pass
    anomaly_threshold   = float(config["anomaly_threshold"])
    uncertain_threshold = anomaly_threshold * 0.5

    is_anomaly   = scores >= anomaly_threshold
    is_uncertain = (~is_anomaly) & (scores >= uncertain_threshold)
    is_normal    = ~(is_anomaly | is_uncertain)
    correct      = pred_labels == true_labels
    accuracy     = float(correct.mean()) * 100

    # ---- Console summary ----
    print("\n" + "=" * 68)
    print(f"  Quick-Test Results  ({n:,} samples,"
          f" fixed threshold={anomaly_threshold})")
    print("=" * 68)
    print(f"  Top-1 Accuracy  : {accuracy:.2f}%  ({int(correct.sum())}/{n})")
    print(f"  Anomaly Score   : "
          f"min={format_pct(scores.min())}  "
          f"mean={format_pct(scores.mean())}  "
          f"max={format_pct(scores.max())}")
    print(f"  ✅  Normal      : {int(is_normal.sum()):>6,}  "
          f"({100 * is_normal.mean():.1f}%)")
    print(f"  ❓  Uncertain   : {int(is_uncertain.sum()):>6,}  "
          f"({100 * is_uncertain.mean():.1f}%)")
    print(f"  ⚠   Anomaly     : {int(is_anomaly.sum()):>6,}  "
          f"({100 * is_anomaly.mean():.1f}%)")
    print("-" * 68)

    # ---- Sample predictions (first 10) ----
    n_show = min(10, n)
    print(f"  Sample predictions (first {n_show}):")
    for i in range(n_show):
        pred_ev = encoder.inverse_transform(
            [int(pred_labels[i])], context=f"qt pred {i}"
        )[0]
        true_ev = encoder.inverse_transform(
            [int(true_labels[i])], context=f"qt true {i}"
        )[0]
        mark  = "✓" if correct[i] else "✗"
        band  = "ANOMALY" if is_anomaly[i] else "UNCERTAIN" if is_uncertain[i] else "NORMAL"
        print(f"  [{mark}] pred={pred_ev:<5}  actual={true_ev:<5}"
              f"  score={format_pct(scores[i])}  [{band_label(band)}]")
    print("=" * 68 + "\n")

    # ---- Save CSV ----
    rows = []
    for i in range(n):
        seq_dec = encoder.inverse_transform(
            sequences[i].tolist(), context=f"qt seq {i}"
        )
        pred_ev = encoder.inverse_transform(
            [int(pred_labels[i])], context=f"qt pred {i}"
        )[0]
        true_ev = encoder.inverse_transform(
            [int(true_labels[i])], context=f"qt true {i}"
        )[0]
        classification = (
            "ANOMALY"   if is_anomaly[i]   else
            "UNCERTAIN" if is_uncertain[i] else
            "NORMAL"
        )
        rows.append({
            "sequence":        " ".join(seq_dec),
            "predicted_event": pred_ev,
            "actual_event":    true_ev,
            "correct":         bool(correct[i]),
            "anomaly_score":   round(float(scores[i]), 6),
            "confidence":      round(1.0 - float(scores[i]), 6),
            "classification":  classification,
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    log.info("Results saved → %s  (%s rows)", output_csv, f"{n:,}")

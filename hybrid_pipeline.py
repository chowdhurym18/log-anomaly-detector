# =============================================================================
# pipeline/hybrid_pipeline.py — Uncertainty-aware hybrid routing.
#
# ⚠ LEGACY (pre-audit) PATH, reachable via `main.py --hybrid`. SUPERSEDED for
# detection by `main.py --detect`. Kept as evidence for the audit finding that
# MSP-confidence routing is the WRONG signal for detection (it routes "is the
# prediction correct?", not "was the actual event surprising?" — see
# outputs/detection_analysis.md, where MSP is the worst block-level detector).
#
# CRITICAL terminology note: the "Confidence (MSP)" used HERE — P(model's OWN
# top-1 predicted event) — is a DIFFERENT quantity from the production detector's
# "confidence" (P of the ACTUAL next event = 1 − surprisal). Do not conflate them.
#
# A second, confidence-first decision path that runs alongside (never replaces)
# the composite-score run_anomaly_pipeline. The flow:
#
#     Log Sequence → GRU → Confidence (MSP) → Normal / Uncertain / Anomaly
#                       → ONLY Uncertain cases go to Llama → Final decision + report
#
# "Confidence" here is the Maximum Softmax Probability (MSP): the probability the
# model assigns to its OWN top-1 predicted next event (see
# evaluation.evaluate.collect_confidence_outputs). High = sure → NORMAL; low =
# unsure → ANOMALY; the ambiguous middle band → UNCERTAIN → escalated to the LLM.
#
# Anti-hallucination safeguards are reused unchanged: explanations go through the
# existing send_to_llm / generate_explanation path, so Python still owns every
# verdict field and the LLM is still confined to one scrubbed narrative field.
# =============================================================================

import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from evaluation.evaluate import collect_confidence_outputs
from preprocessing.encoder import validate_encoded_values
from llm.explanation import send_to_llm

log = logging.getLogger(__name__)


def classify_by_confidence(confidences, normal_thr: float, anomaly_thr: float):
    """Bucket confidences into NORMAL / UNCERTAIN / ANOMALY by two cutoffs.

    Pure and vectorised — the single source of truth for the confidence-banding
    rule, reused by both run_hybrid_pipeline and the evaluation harness (e.g. the
    threshold study, which re-buckets the same confidences with several cutoffs).

    Args:
        confidences : numpy array of per-sequence confidence (MSP) in [0, 1].
        normal_thr  : confidence >= this → NORMAL.
        anomaly_thr : confidence <= this → ANOMALY.
                      Anything strictly in between → UNCERTAIN.

    Returns:
        (is_normal, is_uncertain, is_anomaly) — three boolean numpy masks.
    """
    is_normal    = confidences >= normal_thr
    is_anomaly   = confidences <= anomaly_thr
    is_uncertain = ~(is_normal | is_anomaly)   # strictly between the two cutoffs
    return is_normal, is_uncertain, is_anomaly


def run_hybrid_pipeline(model, X_path, y_path, test_idx, encoder, device, config,
                        event_context: Optional[Dict[str, str]] = None,
                        output_csv: str = "outputs/hybrid_results.csv") -> dict:
    """
    Runs the uncertainty-aware hybrid pipeline over the test set.

    For every test sequence it computes the GRU's self-confidence (MSP), assigns
    NORMAL / UNCERTAIN / ANOMALY by two fixed confidence cutoffs, sends ONLY the
    UNCERTAIN cases to the LLM, prints a metrics block, and writes a per-sequence
    CSV report.

    Args:
        model         : Trained GRUAnomalyDetector (eval mode).
        X_path        : Path to memmap X array.
        y_path        : Path to memmap y array.
        test_idx      : Test-set index array (already subset-limited if requested).
        encoder       : Fitted StreamingLabelEncoder.
        device        : Compute device.
        config        : CONFIG dict (reads the confidence_*_threshold keys).
        event_context : Optional EventId → template mapping for explanations.
        output_csv    : Destination CSV for the per-sequence report.

    Returns:
        A metrics dict (counts, percentages, average confidence, runtime) — handy
        for logging and for any caller that wants the numbers programmatically.
    """
    t0 = time.perf_counter()

    # ---- 1. Score every test sequence by self-confidence (MSP) ----
    log.info("Running uncertainty-aware hybrid pipeline ...")
    confidences, pred_labels, true_labels, sequences = collect_confidence_outputs(
        model, X_path, y_path, test_idx, encoder, device, config, name="hybrid_test"
    )

    validate_encoded_values("hybrid pipeline sequences", sequences, encoder)
    validate_encoded_values("hybrid pipeline true labels", true_labels, encoder)
    validate_encoded_values("hybrid pipeline model predictions", pred_labels, encoder)

    # ---- 2. Confidence-based classification (vectorised) ----
    normal_thr  = float(config["confidence_normal_threshold"])    # e.g. 0.70
    anomaly_thr = float(config["confidence_anomaly_threshold"])   # e.g. 0.40

    is_normal, is_uncertain, is_anomaly = classify_by_confidence(
        confidences, normal_thr, anomaly_thr
    )

    n_total     = int(len(confidences))
    n_normal    = int(is_normal.sum())
    n_anomaly   = int(is_anomaly.sum())
    n_uncertain = int(is_uncertain.sum())
    avg_conf    = float(confidences.mean()) if n_total else 0.0

    # ---- 3. Route ONLY uncertain cases to Llama ----
    # Order uncertain cases by confidence ASCENDING so, if we have to cap how many
    # the LLM explains, the least-confident (most in-need) cases go first.
    uncertain_idx = np.where(is_uncertain)[0]
    uncertain_idx = uncertain_idx[np.argsort(confidences[uncertain_idx])]

    max_expl = config.get("max_llm_explanations")
    routed_idx = uncertain_idx
    if max_expl and len(routed_idx) > max_expl:
        routed_idx = routed_idx[:max_expl]
        log.info("Routing the %d lowest-confidence of %d uncertain cases to the "
                 "LLM (config 'max_llm_explanations').", max_expl, n_uncertain)

    # `routed_to_llama` flag, aligned to every test row.
    routed_mask = np.zeros(n_total, dtype=bool)
    routed_mask[routed_idx] = True

    uncertain_cases = [
        {
            "sequence": encoder.inverse_transform(
                sequences[i], context=f"hybrid uncertain row {i}"
            ),
            "predicted_event": encoder.inverse_transform(
                [int(pred_labels[i])], context=f"hybrid predicted label row {i}"
            )[0],
            "actual_event": encoder.inverse_transform(
                [int(true_labels[i])], context=f"hybrid true label row {i}"
            )[0],
            "confidence":     float(confidences[i]),
            # No composite score in this path; derive a [0,1] anomaly value from
            # confidence so the report's "Anomaly Score" field is populated, and
            # the existing convention confidence = 1 - anomaly_score holds.
            "anomaly_score":  float(1.0 - confidences[i]),
            "classification": "UNCERTAIN",
        }
        for i in routed_idx
    ]

    if uncertain_cases:
        send_to_llm(uncertain_cases, event_context)
    else:
        log.info("No uncertain cases to route to the LLM.")

    # ---- 4. Metrics block ----
    runtime = time.perf_counter() - t0
    pct_routed = (100.0 * len(routed_idx) / n_total) if n_total else 0.0
    seqs_per_sec = (n_total / runtime) if runtime > 0 else 0.0

    print("\n" + "=" * 68)
    print("  Hybrid Uncertainty-Aware Results")
    print(f"  (confidence = max softmax prob; NORMAL >= {normal_thr}, "
          f"ANOMALY <= {anomaly_thr})")
    print("=" * 68)
    print(f"  Total sequences   : {n_total:,}")
    print(f"  ✅  Normal        : {n_normal:>7,}  ({100 * is_normal.mean():.1f}%)")
    print(f"  ❓  Uncertain     : {n_uncertain:>7,}  ({100 * is_uncertain.mean():.1f}%)")
    print(f"  ⚠   Anomaly       : {n_anomaly:>7,}  ({100 * is_anomaly.mean():.1f}%)")
    print(f"  → Routed to Llama : {len(routed_idx):>7,}  ({pct_routed:.1f}%)")
    print(f"  Average confidence: {avg_conf:.4f}")
    print(f"  Runtime           : {runtime:.2f}s  ({seqs_per_sec:,.0f} seq/s)")
    print("=" * 68 + "\n")

    log.info("Hybrid results — normal=%d uncertain=%d anomaly=%d  routed_to_llama=%d "
             "(%.1f%%)  avg_conf=%.4f  runtime=%.2fs",
             n_normal, n_uncertain, n_anomaly, len(routed_idx), pct_routed,
             avg_conf, runtime)

    # ---- 5. Per-sequence CSV report (one row per test sequence) ----
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_total):
        seq_dec = encoder.inverse_transform(
            sequences[i].tolist(), context=f"hybrid csv seq {i}"
        )
        pred_ev = encoder.inverse_transform(
            [int(pred_labels[i])], context=f"hybrid csv pred {i}"
        )[0]
        true_ev = encoder.inverse_transform(
            [int(true_labels[i])], context=f"hybrid csv true {i}"
        )[0]
        classification = (
            "NORMAL"    if is_normal[i]    else
            "ANOMALY"   if is_anomaly[i]   else
            "UNCERTAIN"
        )
        rows.append({
            "sequence":         " ".join(seq_dec),
            "predicted_event":  pred_ev,
            "actual_event":     true_ev,
            "confidence_score": round(float(confidences[i]), 6),
            "classification":   classification,
            "routed_to_llama":  bool(routed_mask[i]),
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    log.info("Hybrid results saved → %s  (%s rows)", output_csv, f"{n_total:,}")

    return {
        "total":            n_total,
        "normal":           n_normal,
        "uncertain":        n_uncertain,
        "anomaly":          n_anomaly,
        "routed_to_llama":  int(len(routed_idx)),
        "pct_routed":       pct_routed,
        "average_confidence": avg_conf,
        "runtime_seconds":  runtime,
    }

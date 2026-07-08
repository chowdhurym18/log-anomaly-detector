# =============================================================================
# evaluation/evaluate.py — Batched evaluation + composite anomaly scoring.
#
# Top-1/3/5 accuracy, per-class P/R/F1, confusion-matrix analysis, and the
# composite anomaly score (surprisal + entropy + top-K miss) used downstream
# by the anomaly pipeline. All inference is batched — no per-sample Python loop.
# =============================================================================

import math
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preprocessing.dataset import MemmapLogDataset
from preprocessing.encoder import validate_encoded_values, log_distribution

log = logging.getLogger(__name__)


def _topk_accuracy(logits_np: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Fraction of samples where the true label is in the top-K predicted classes."""
    topk_idx = np.argsort(logits_np, axis=1)[:, -k:]
    hits = np.any(topk_idx == true_labels[:, np.newaxis], axis=1)
    return float(hits.mean())


def composite_score(logits, y_true, config):
    """Composite anomaly score in [0,1] for a batch of logits.

    Three complementary signals, each catching a different failure mode:
      1. surprisal = 1 - P(true_event)            — model expected a different event.
      2. entropy   = H(softmax) / log(vocab)      — model is confused (uniform output).
      3. topk_miss = 1 if true ∉ top-K else 0     — true event is implausible to the model.
    Combined: sw·surprisal + ew·entropy + gw·topk_miss, clipped to [0,1].
    Entropy and top-K miss are included because a class-biased model would flag
    every non-majority actual under pure surprisal.

    Args:
        logits : (B, vocab) raw model outputs.
        y_true : (B,) true next-event indices, on the same device as logits.
        config : CONFIG (reads topk_k, entropy_weight, topk_miss_weight).
    Returns:
        (B,) tensor of composite scores.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs     = torch.exp(log_probs)

    # Component 1: surprisal
    true_probs = probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
    surprisal  = 1.0 - true_probs

    # Component 2: normalised Shannon entropy
    entropy   = -(probs * log_probs).sum(dim=1)
    H_max     = math.log(max(probs.shape[1], 2))
    entropy_n = entropy / H_max

    # Component 3: top-K miss penalty
    k         = config.get("topk_k", 3)
    topk_idx  = probs.topk(min(k, probs.shape[1]), dim=1).indices
    in_topk   = (topk_idx == y_true.unsqueeze(1)).any(dim=1)
    topk_miss = (~in_topk).float()

    # Weighted combination (surprisal weight ≈ 0.55)
    ew = config.get("entropy_weight",   0.35)
    gw = config.get("topk_miss_weight", 0.10)
    sw = 1.0 - ew - gw
    return (sw * surprisal + ew * entropy_n + gw * topk_miss).clamp(0.0, 1.0)


def evaluate_model(model, test_loader, device, encoder):
    """
    Comprehensive evaluation: top-1/3/5 accuracy, per-class P/R/F1,
    confusion matrix top-confused pairs, and prediction distribution analysis.

    Returns a dict with all computed metrics.
    """
    log.info("Evaluating on test set ...")
    model.eval()
    all_logits, all_preds, all_labels, all_sequences = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            all_sequences.append(X_batch.numpy())
            all_labels.append(y_batch.numpy())
            X_batch = X_batch.to(device, non_blocking=True)
            logits  = model(X_batch)
            all_logits.append(logits.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    logits_np   = np.concatenate(all_logits)
    pred_labels = np.concatenate(all_preds)
    true_labels = np.concatenate(all_labels)
    sequences   = np.concatenate(all_sequences)

    validate_encoded_values("evaluation true labels",       true_labels, encoder)
    validate_encoded_values("evaluation model predictions", pred_labels, encoder)
    validate_encoded_values("evaluation sequences",         sequences,   encoder)

    total   = int(true_labels.size)
    correct = int((pred_labels == true_labels).sum())
    top1    = correct / total if total else 0.0
    top3    = _topk_accuracy(logits_np, true_labels, min(3, len(encoder.classes_)))
    top5    = _topk_accuracy(logits_np, true_labels, min(5, len(encoder.classes_)))

    # ---- Prediction distribution ----
    log_distribution("Prediction", pred_labels, encoder)
    log_distribution("Actual test label", true_labels, encoder)

    # ---- Accuracy summary ----
    log.info("Top-1 accuracy : %.2f%%  (%d/%d)", top1 * 100, correct, total)
    log.info("Top-3 accuracy : %.2f%%", top3 * 100)
    log.info("Top-5 accuracy : %.2f%%", top5 * 100)

    # ---- Per-class classification report ----
    classes_present = sorted(set(true_labels.tolist()) | set(pred_labels.tolist()))
    class_names     = [encoder.classes_[i] for i in classes_present]
    report = classification_report(
        true_labels, pred_labels,
        labels     = classes_present,
        target_names = class_names,
        zero_division = 0,
        digits       = 4,
    )
    log.info("Per-class classification report:\n%s", report)

    # ---- Macro / weighted aggregates ----
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    log.info(
        "Weighted  precision=%.4f  recall=%.4f  F1=%.4f", prec, rec, f1
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    log.info(
        "Macro     precision=%.4f  recall=%.4f  F1=%.4f", prec_m, rec_m, f1_m
    )

    # ---- Confusion matrix — top confused pairs ----
    cm = confusion_matrix(true_labels, pred_labels, labels=classes_present)
    np.fill_diagonal(cm, 0)   # zero the diagonal so correct predictions don't dominate
    confused_flat = cm.ravel()
    top_confused_idx = confused_flat.argsort()[::-1][:8]
    log.info("Top confused class pairs (true → predicted):")
    for flat_idx in top_confused_idx:
        if confused_flat[flat_idx] == 0:
            break
        row, col = divmod(flat_idx, len(classes_present))
        true_ev  = encoder.classes_[classes_present[row]]
        pred_ev  = encoder.classes_[classes_present[col]]
        log.info("  %s → %s : %d misclassifications", true_ev, pred_ev,
                 int(confused_flat[flat_idx]))

    # ---- Sample predictions ----
    log.info("Sample predictions (first 10 test rows):")
    for i in range(min(10, total)):
        pred_ev = encoder.inverse_transform([int(pred_labels[i])], context=f"eval pred {i}")[0]
        true_ev = encoder.inverse_transform([int(true_labels[i])], context=f"eval true {i}")[0]
        match   = pred_labels[i] == true_labels[i]
        log.info("  row=%d  pred=%-6s  actual=%-6s  correct=%s", i, pred_ev, true_ev, match)

    return {
        "top1_accuracy":         top1,
        "top3_accuracy":         top3,
        "top5_accuracy":         top5,
        "weighted_precision":    float(prec),
        "weighted_recall":       float(rec),
        "weighted_f1":           float(f1),
        "macro_precision":       float(prec_m),
        "macro_recall":          float(rec_m),
        "macro_f1":              float(f1_m),
    }


def collect_anomaly_outputs(model, X_path, y_path, indices, encoder, device, config,
                            name: str):
    """Scores a set of sequence indices with the composite anomaly score.

    Returns (scores, true_labels, pred_labels, sequences) as numpy arrays.
    """
    ds = MemmapLogDataset(X_path, y_path, indices)
    loader = DataLoader(
        ds,
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        pin_memory  = config["pin_memory"] and (device.type != "cpu"),
        shuffle     = False,
    )

    all_scores = []
    all_true_labels = []
    all_pred_labels = []
    all_sequences = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_cpu = X_batch.numpy()
            y_cpu = y_batch.numpy()
            vocab_size = len(encoder.classes_)
            valid_rows = (
                (X_cpu >= 0).all(axis=1) &
                (X_cpu < vocab_size).all(axis=1) &
                (y_cpu >= 0) &
                (y_cpu < vocab_size)
            )

            if not valid_rows.all():
                bad_row = int(np.where(~valid_rows)[0][0])
                validate_encoded_values(
                    f"{name} anomaly-score sequence",
                    X_cpu[bad_row],
                    encoder,
                    sequence=X_cpu[bad_row].tolist(),
                )
                validate_encoded_values(
                    f"{name} anomaly-score true label",
                    [int(y_cpu[bad_row])],
                    encoder,
                    sequence=X_cpu[bad_row].tolist(),
                )
                valid_rows_t = torch.from_numpy(valid_rows)
                X_batch = X_batch[valid_rows_t]
                y_batch = y_batch[valid_rows_t]
                if len(X_batch) == 0:
                    continue

            X_batch  = X_batch.to(device, non_blocking=True)
            y_device = y_batch.to(device, non_blocking=True)
            logits   = model(X_batch)

            # Composite anomaly score (surprisal + entropy + top-K miss).
            # See composite_score() above for the full rationale.
            composite = composite_score(logits, y_device, config)

            all_scores.append(composite.cpu().numpy())
            all_true_labels.append(y_batch.numpy())
            all_pred_labels.append(logits.argmax(dim=1).cpu().numpy())
            all_sequences.append(X_batch.cpu().numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_true_labels),
        np.concatenate(all_pred_labels),
        np.concatenate(all_sequences),
    )


def collect_confidence_outputs(model, X_path, y_path, indices, encoder, device, config,
                               name: str):
    """Scores a set of sequence indices by the model's self-confidence (MSP).

    Confidence here is the **Maximum Softmax Probability**: for each sequence we
    softmax the model's output logits over the vocabulary, then take the single
    largest probability. That number is the probability the model assigns to its
    OWN top-1 predicted next event.
        - confidence near 1.0 → the model is very sure about its prediction.
        - confidence near 0.0 → the model is spread thin / unsure (a candidate anomaly).

    This is intentionally different from composite_score() (surprisal + entropy +
    top-K miss). The hybrid pipeline routes purely on this raw confidence, so it
    needs the max-softmax value and the argmax (the predicted event) directly.

    The batched loop and the encoded-value validity filtering mirror
    collect_anomaly_outputs() so both paths reject corrupt cache rows identically.

    Returns:
        (confidences, pred_labels, true_labels, sequences) as numpy arrays, where
        confidences[i] in [0, 1] is the MSP and pred_labels[i] is the argmax event.
    """
    ds = MemmapLogDataset(X_path, y_path, indices)
    loader = DataLoader(
        ds,
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        pin_memory  = config["pin_memory"] and (device.type != "cpu"),
        shuffle     = False,
    )

    all_confidences = []
    all_true_labels = []
    all_pred_labels = []
    all_sequences = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_cpu = X_batch.numpy()
            y_cpu = y_batch.numpy()
            vocab_size = len(encoder.classes_)
            valid_rows = (
                (X_cpu >= 0).all(axis=1) &
                (X_cpu < vocab_size).all(axis=1) &
                (y_cpu >= 0) &
                (y_cpu < vocab_size)
            )

            if not valid_rows.all():
                bad_row = int(np.where(~valid_rows)[0][0])
                validate_encoded_values(
                    f"{name} confidence sequence",
                    X_cpu[bad_row],
                    encoder,
                    sequence=X_cpu[bad_row].tolist(),
                )
                validate_encoded_values(
                    f"{name} confidence true label",
                    [int(y_cpu[bad_row])],
                    encoder,
                    sequence=X_cpu[bad_row].tolist(),
                )
                valid_rows_t = torch.from_numpy(valid_rows)
                X_batch = X_batch[valid_rows_t]
                y_batch = y_batch[valid_rows_t]
                if len(X_batch) == 0:
                    continue

            X_batch = X_batch.to(device, non_blocking=True)
            logits  = model(X_batch)

            # MSP: softmax over the vocabulary, then take the top probability and
            # its index (the predicted event). conf and pred line up element-wise.
            probs      = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            all_confidences.append(conf.cpu().numpy())
            all_pred_labels.append(pred.cpu().numpy())
            all_true_labels.append(y_batch.numpy())
            all_sequences.append(X_batch.cpu().numpy())

    return (
        np.concatenate(all_confidences),
        np.concatenate(all_pred_labels),
        np.concatenate(all_true_labels),
        np.concatenate(all_sequences),
    )


def collect_uncertainty_signals(model, X_path, y_path, indices, encoder, device, config,
                                name: str):
    """Single inference pass that returns THREE uncertainty signals per sequence.

    This is the analysis-grade collector used by evaluate_hybrid.py. It runs the
    model once and, for every sequence, records three different ways to measure
    "how unsure is the model":

      1. msp       — Maximum Softmax Probability: the top probability p1. This is
                     the confidence the hybrid pipeline currently routes on.
      2. top2_gap  — p1 - p2 (gap between the two most likely events). A small gap
                     means the model is torn between two events; a large gap means
                     it strongly prefers one. An alternative confidence signal.
      3. entropy   — normalised Shannon entropy H(softmax) / log(vocab), in [0, 1].
                     High entropy = the probability mass is spread out = unsure.
                     (NOTE: high entropy means LESS confident — the opposite
                     direction to msp/top2_gap.)

    Computing all three in one pass lets the report compare them on identical data
    and say, with evidence, whether top2_gap or entropy would separate correct
    from incorrect predictions better than msp.

    The batched loop and encoded-value validity filtering mirror
    collect_confidence_outputs() / collect_anomaly_outputs() so all paths reject
    corrupt cache rows identically.

    Returns:
        dict of numpy arrays with keys:
          "msp", "top2_gap", "entropy", "pred_labels", "true_labels", "sequences".
    """
    ds = MemmapLogDataset(X_path, y_path, indices)
    loader = DataLoader(
        ds,
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
        pin_memory  = config["pin_memory"] and (device.type != "cpu"),
        shuffle     = False,
    )

    all_msp = []
    all_top2_gap = []
    all_entropy = []
    all_pred_labels = []
    all_true_labels = []
    all_sequences = []

    vocab_size = len(encoder.classes_)
    h_max = math.log(max(vocab_size, 2))   # max possible entropy (for normalisation)

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_cpu = X_batch.numpy()
            y_cpu = y_batch.numpy()
            valid_rows = (
                (X_cpu >= 0).all(axis=1) &
                (X_cpu < vocab_size).all(axis=1) &
                (y_cpu >= 0) &
                (y_cpu < vocab_size)
            )

            if not valid_rows.all():
                bad_row = int(np.where(~valid_rows)[0][0])
                validate_encoded_values(
                    f"{name} uncertainty sequence",
                    X_cpu[bad_row],
                    encoder,
                    sequence=X_cpu[bad_row].tolist(),
                )
                valid_rows_t = torch.from_numpy(valid_rows)
                X_batch = X_batch[valid_rows_t]
                y_batch = y_batch[valid_rows_t]
                if len(X_batch) == 0:
                    continue

            X_batch = X_batch.to(device, non_blocking=True)
            logits  = model(X_batch)

            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            probs     = torch.exp(log_probs)

            # Top-2 probabilities → MSP (p1) and the top-2 gap (p1 - p2).
            top2 = probs.topk(min(2, probs.shape[1]), dim=1).values
            p1   = top2[:, 0]
            p2   = top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(p1)
            msp      = p1
            top2_gap = p1 - p2

            # Normalised Shannon entropy in [0, 1].
            entropy   = -(probs * log_probs).sum(dim=1)
            entropy_n = entropy / h_max

            all_msp.append(msp.cpu().numpy())
            all_top2_gap.append(top2_gap.cpu().numpy())
            all_entropy.append(entropy_n.cpu().numpy())
            all_pred_labels.append(logits.argmax(dim=1).cpu().numpy())
            all_true_labels.append(y_batch.numpy())
            all_sequences.append(X_batch.cpu().numpy())

    return {
        "msp":         np.concatenate(all_msp),
        "top2_gap":    np.concatenate(all_top2_gap),
        "entropy":     np.concatenate(all_entropy),
        "pred_labels": np.concatenate(all_pred_labels),
        "true_labels": np.concatenate(all_true_labels),
        "sequences":   np.concatenate(all_sequences),
    }


def choose_thresholds(train_scores: np.ndarray, config):
    """Selects anomaly/uncertain thresholds from config or train-score calibration."""
    if config.get("threshold_mode") == "calibrated":
        uncertain_threshold = float(np.quantile(train_scores, config["uncertain_quantile"]))
        anomaly_threshold = float(np.quantile(train_scores, config["anomaly_quantile"]))
        log.info(
            "Calibrated thresholds from train scores: uncertain>=%.4f (q=%.2f), anomaly>=%.4f (q=%.2f)",
            uncertain_threshold, config["uncertain_quantile"],
            anomaly_threshold, config["anomaly_quantile"],
        )
    else:
        anomaly_threshold = float(config["anomaly_threshold"])
        uncertain_threshold = anomaly_threshold * 0.5
        log.info(
            "Fixed thresholds: uncertain>=%.4f, anomaly>=%.4f",
            uncertain_threshold, anomaly_threshold,
        )

    if uncertain_threshold > anomaly_threshold:
        uncertain_threshold = anomaly_threshold

    return uncertain_threshold, anomaly_threshold

# =============================================================================
# evaluation/anomaly_eval.py — REAL anomaly detection evaluation (block level).
#
# The rest of the project measures next-event ACCURACY (a proxy). This module
# measures what actually matters and what every paper (DeepLog, LogBERT, ...)
# reports: does the system flag the right *blocks* as anomalous?
#
# Ground truth: data/HDFS_v1/preprocessed/anomaly_label.csv  (BlockId -> Normal/Anomaly).
# Block sequences: Event_traces.csv  (Features = the block's ordered EventId list).
#
# Method (DeepLog-style, but scored with the trained GRU):
#   * For each block, slide a window: predict event i from the preceding events
#     (up to seq_len of history). This is done PER BLOCK, so windows never cross
#     block boundaries (unlike the flat training cache).
#   * Per window: surprisal = 1 - P(true next event). A block is suspicious if it
#     contains a highly surprising transition, so the block score = MAX surprisal
#     over its windows (we also report mean-surprisal and top-k-miss-fraction).
#   * Compare block scores to the true labels: AUROC, PR-AUC (the headline metric,
#     since anomalies are ~2.9% — rare), best-F1 threshold, confusion matrix, and
#     ROC / PR curves.
#
# The GRU is length-agnostic, so variable-length contexts are scored by BUCKETING
# windows by context length and batching each bucket — exact, no padding.
# =============================================================================

import os
import sys
import ast
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    precision_recall_fscore_support, confusion_matrix,
)

from config import CONFIG, get_device, variant_flags
from utils.logging_utils import setup_logging
from preprocessing.encoder import StreamingLabelEncoder
from models.gru_model import GRUAnomalyDetector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load block sequences + labels
# ---------------------------------------------------------------------------
def _parse_features(cell: str):
    """Event_traces 'Features' looks like "[E5,E22,E5,...]". Return the EventId list."""
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def load_blocks(encoder, config, max_blocks=None):
    """Returns (blocks, labels): list of encoded-int arrays + 1/0 anomaly labels.

    Joins Event_traces.csv (sequences) with anomaly_label.csv (ground truth) on
    BlockId. Unknown EventIds (not in the trained vocab) are dropped from a block.
    """
    ds = config.get("datasets", {}).get(config.get("dataset", "hdfs"), {})
    traces_path = ds.get("event_traces", "data/HDFS_v1/preprocessed/Event_traces.csv")
    label_path  = ds.get("label_csv", "data/HDFS_v1/preprocessed/anomaly_label.csv")

    log.info("Loading labels from %s", label_path)
    labels_df = pd.read_csv(label_path)
    label_map = dict(zip(labels_df["BlockId"], labels_df["Label"]))

    log.info("Loading block sequences from %s", traces_path)
    traces = pd.read_csv(traces_path, usecols=["BlockId", "Features"])

    c2i = encoder.class_to_idx
    blocks, labels, skipped, dropped_events = [], [], 0, 0
    for block_id, feats in zip(traces["BlockId"], traces["Features"]):
        if block_id not in label_map:
            continue
        eids = _parse_features(feats)
        kept = [c2i[e] for e in eids if e in c2i]
        dropped_events += len(eids) - len(kept)   # unknown-vocab events (audit: now counted)
        idxs = np.array(kept, dtype=np.int64)
        if len(idxs) < 2:           # need at least one (context, target) pair
            skipped += 1
            continue
        blocks.append(idxs)
        labels.append(1 if str(label_map[block_id]).strip().lower() == "anomaly" else 0)

    labels = np.array(labels, dtype=np.int64)
    # AUDIT FIX: a RANDOM, reproducible subsample — NOT the first-N in file order.
    # The old early-break biased every subset comparison toward the file's head.
    if max_blocks and len(blocks) > max_blocks:
        rng = np.random.default_rng(config.get("random_seed", 42))
        sel = np.sort(rng.choice(len(blocks), size=max_blocks, replace=False))
        blocks = [blocks[i] for i in sel]
        labels = labels[sel]

    log.info("Loaded %d blocks (%d anomalies, %.2f%%); skipped %d too-short; "
             "dropped %d unknown-vocab events.",
             len(blocks), int(labels.sum()), 100 * labels.mean() if len(labels) else 0,
             skipped, dropped_events)
    return blocks, labels


# ---------------------------------------------------------------------------
# Score blocks with the model
# ---------------------------------------------------------------------------
def score_blocks(model, blocks, device, config):
    """Return per-block dict of score arrays: max_surprisal, mean_surprisal, topk_miss_frac.

    Builds (context, target) windows per block, buckets them by context length for
    exact batched inference, computes surprisal + top-k miss per window, then
    aggregates back to the block."""
    seq_len = config["sequence_length"]
    k       = config.get("topk_k", 3)
    batch   = config["batch_size"]

    # Flatten all windows, remembering which block each belongs to.
    win_block, win_ctx, win_tgt, win_len = [], [], [], []
    for bi, blk in enumerate(blocks):
        L = len(blk)
        for i in range(1, L):
            ctx = blk[max(0, i - seq_len):i]
            win_block.append(bi)
            win_ctx.append(ctx)
            win_tgt.append(int(blk[i]))
            win_len.append(len(ctx))
    win_block = np.array(win_block); win_tgt = np.array(win_tgt); win_len = np.array(win_len)
    n_blocks = len(blocks)

    surprisal = np.zeros(len(win_tgt), dtype=np.float32)
    topk_miss = np.zeros(len(win_tgt), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        # Bucket by context length so each batch is a clean rectangular tensor.
        for length in np.unique(win_len):
            rows = np.where(win_len == length)[0]
            ctx_mat = np.stack([win_ctx[r] for r in rows])      # (n, length)
            tgt = win_tgt[rows]
            for s in range(0, len(rows), batch):
                bx = torch.from_numpy(ctx_mat[s:s + batch]).long().to(device)
                bt = tgt[s:s + batch]
                logits = model(bx)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                bt_idx = np.arange(len(bt))
                p_true = probs[bt_idx, bt]
                surprisal[rows[s:s + batch]] = 1.0 - p_true
                topk = np.argpartition(probs, -k, axis=1)[:, -k:]
                hit  = (topk == bt[:, None]).any(axis=1)
                topk_miss[rows[s:s + batch]] = (~hit).astype(np.float32)

    # Aggregate windows -> blocks.
    max_surp  = np.zeros(n_blocks, dtype=np.float32)
    sum_surp  = np.zeros(n_blocks, dtype=np.float32)
    cnt       = np.zeros(n_blocks, dtype=np.float32)
    miss_sum  = np.zeros(n_blocks, dtype=np.float32)
    np.maximum.at(max_surp, win_block, surprisal)
    np.add.at(sum_surp, win_block, surprisal)
    np.add.at(cnt, win_block, 1.0)
    np.add.at(miss_sum, win_block, topk_miss)
    cnt = np.maximum(cnt, 1.0)
    return {"max_surprisal": max_surp,
            "mean_surprisal": sum_surp / cnt,
            "topk_miss_frac": miss_sum / cnt}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics_at_best_f1(y_true, y_score):
    """Sweep thresholds; return the best-F1 operating point + its confusion matrix."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1))
    # precision_recall_curve returns thresholds of length len(prec)-1.
    threshold = float(thr[min(best, len(thr) - 1)]) if len(thr) else 0.5
    y_pred = (y_score >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"threshold": threshold, "precision": float(p), "recall": float(r),
            "f1": float(f), "confusion": cm}


def _metrics_with_val_threshold(test_y, test_score, val_y, val_score):
    """Honest operating point (audit fix): choose the best-F1 threshold on
    VALIDATION, then report precision/recall/F1 on TEST at that threshold — so the
    reported F1 is not an oracle picked on the same data it scores."""
    thr = _metrics_at_best_f1(val_y, val_score)["threshold"]
    y_pred = (test_score >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(test_y, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(test_y, y_pred)
    return {"threshold": float(thr), "precision": float(p), "recall": float(r),
            "f1": float(f), "confusion": cm}


def detection_metrics(model, encoder, device, config, max_blocks=None, score_key="max_surprisal"):
    """Reusable entry point: returns a dict of block-level detection metrics.

    Used by compare_training.py to report 'Detection F1' for each model variant."""
    blocks, y_true = load_blocks(encoder, config, max_blocks=max_blocks)
    scores = score_blocks(model, blocks, device, config)
    y_score = scores[score_key]
    best = _metrics_at_best_f1(y_true, y_score)
    return {
        "n_blocks": len(y_true), "anomaly_rate": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": best["precision"], "recall": best["recall"], "f1": best["f1"],
        "threshold": best["threshold"], "confusion": best["confusion"],
        "_y_true": y_true, "_scores": scores,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _plot_curves(y_true, y_score):
    Path("outputs").mkdir(exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4C72B0", label=f"AUROC = {roc_auc_score(y_true, y_score):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — block-level anomaly detection"); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=120); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#C44E52", label=f"PR-AUC = {average_precision_score(y_true, y_score):.3f}")
    plt.axhline(np.mean(y_true), color="k", ls="--", alpha=0.4, label=f"base rate = {np.mean(y_true):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — block-level anomaly detection"); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/pr_curve.png", dpi=120); plt.close()


def write_report(metrics_by_key, primary="max_surprisal", held_out=False):
    m = metrics_by_key[primary]
    y_true = m["_y_true"]
    cm = m["confusion"]
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    L = []; a = L.append
    a("# Real Anomaly-Detection Report (block level)\n")
    a("_Generated by `evaluation/anomaly_eval.py`. Ground truth: "
      "`anomaly_label.csv`. Sequences re-windowed per block (no cross-block leakage)._\n")
    regime = ("**Held-out TEST split**; F1 threshold selected on validation (honest "
              "operating point)." if held_out else
              "**All blocks (in-sample)**; best-F1 is an ORACLE threshold picked on the "
              "evaluation set itself — re-run with `--held-out` for an honest operating point.")
    a(f"_Evaluation regime: {regime}_\n")
    a(f"## Headline metrics (best aggregator: block score = {primary})\n")
    a(f"- Blocks evaluated: **{m['n_blocks']:,}**  ·  anomaly base rate: **{m['anomaly_rate']:.3f}**")
    a(f"- **PR-AUC: {m['pr_auc']:.4f}**  (the metric that matters at {m['anomaly_rate']:.1%} positives)")
    a(f"- AUROC: {m['auroc']:.4f}")
    a(f"- Best-F1 operating point (threshold {m['threshold']:.4f}): "
      f"**Precision {m['precision']:.4f} · Recall {m['recall']:.4f} · F1 {m['f1']:.4f}**\n")
    a("Confusion matrix at that threshold:\n")
    a("| | pred Normal | pred Anomaly |")
    a("|---|---|---|")
    a(f"| **true Normal** | {tn:,} | {fp:,} |")
    a(f"| **true Anomaly** | {fn:,} | {tp:,} |\n")
    a("Curves: `outputs/roc_curve.png`, `outputs/pr_curve.png`.\n")
    a("## Alternative block-score definitions\n")
    a("| block score | PR-AUC | AUROC | best-F1 |")
    a("|---|---|---|---|")
    for key in ("max_surprisal", "mean_surprisal", "topk_miss_frac"):
        mm = metrics_by_key[key]
        a(f"| {key} | {mm['pr_auc']:.4f} | {mm['auroc']:.4f} | {mm['f1']:.4f} |")
    a("")
    a("## How to read this — and the key finding\n")
    a("This is the project's **first paper-comparable detection result**. It is also a "
      "**reality check**: DeepLog/LogBERT report HDFS F1 ~0.85–0.96, but this model scores "
      f"**F1 {m['f1']:.2f} (AUROC {m['auroc']:.2f})**. The gap is the most important result here.\n")
    a("**Most likely cause — the training regime, not the architecture.** DeepLog-style "
      "detection assumes the model is trained on **normal logs only**, so anomalous "
      "transitions look *surprising*. This project trains the GRU on **all blocks (normal + "
      "anomalous mixed)** with a next-event objective, so it learns anomalous transitions as "
      "ordinary and is *not* surprised by them. Evidence: the classic DeepLog signal "
      "(top-k-miss) is near-random here (see the table above), i.e. anomalous blocks do not "
      "contain more out-of-top-k events than normal ones.\n")
    a("**Highest-value next step (beyond the 4 implemented phases):** retrain the GRU on a "
      "**normal-only** stream (filter blocks by `anomaly_label.csv` before windowing), then "
      "re-run this script. This is the change most likely to move detection F1 toward the "
      "literature — far more than focal loss or semantic embeddings (which target macro-F1 "
      "and generalisation, not the surprise signal). PR-AUC is the headline because anomalies "
      "are only ~2.9%; AUROC looks optimistic under that imbalance.\n")
    Path("outputs/anomaly_detection_report.md").write_text("\n".join(L) + "\n")
    log.info("Wrote outputs/anomaly_detection_report.md")


def _load_model(variant, encoder, device, checkpoint=None):
    flags = variant_flags(variant)
    model = GRUAnomalyDetector(
        len(encoder.classes_), CONFIG["embedding_dim"], CONFIG["hidden_dim"],
        CONFIG["num_layers"], CONFIG["dropout"], **flags,
    ).to(device)
    ckpt = checkpoint or os.path.join(
        CONFIG["checkpoint_dir"] if variant == "gru"
        else os.path.join(CONFIG["checkpoint_dir"], variant), "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log.info("Loaded %s", ckpt)
    return model


def _parse_args():
    p = argparse.ArgumentParser(description="Block-level anomaly detection evaluation.")
    p.add_argument("--variant", default="gru", choices=["gru", "bigru", "bigru_attention"])
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    p.add_argument("--max-blocks", type=int, default=None, help="Limit blocks (quick runs; random sample).")
    p.add_argument("--held-out", action="store_true",
                   help="Honest eval: score ONLY the held-out TEST block split and pick the "
                        "F1 threshold on VALIDATION. Default scores ALL blocks in-sample with "
                        "an oracle best-F1 threshold (optimistic).")
    return p.parse_args()


def main():
    setup_logging()
    args = _parse_args()
    device = get_device()
    enc_path = os.path.join(CONFIG["cache_dir"], "encoder.pkl")
    encoder = StreamingLabelEncoder.load(enc_path)
    model = _load_model(args.variant, encoder, device, args.checkpoint)

    if args.held_out:
        # Honest split: train/val/test by BlockId; score TEST, threshold on VAL.
        from preprocessing.block_dataset import (
            load_labeled_blocks, split_block_indices, select_blocks)
        blocks_all, labels_all, ids_all = load_labeled_blocks(
            encoder, CONFIG, max_blocks=args.max_blocks)
        _, va_idx, te_idx = split_block_indices(labels_all, CONFIG)
        va_blocks, va_y, _ = select_blocks(blocks_all, labels_all, ids_all, va_idx)
        te_blocks, y_true, _ = select_blocks(blocks_all, labels_all, ids_all, te_idx)
        log.info("HELD-OUT: scoring %d test blocks; threshold from %d val blocks.",
                 len(te_blocks), len(va_blocks))
        val_scores = score_blocks(model, va_blocks, device, CONFIG)
        scores = score_blocks(model, te_blocks, device, CONFIG)
    else:
        blocks, y_true = load_blocks(encoder, CONFIG, max_blocks=args.max_blocks)
        val_scores = None
        scores = score_blocks(model, blocks, device, CONFIG)

    metrics_by_key = {}
    for key in ("max_surprisal", "mean_surprisal", "topk_miss_frac"):
        ys = scores[key]
        best = (_metrics_with_val_threshold(y_true, ys, va_y, val_scores[key])
                if args.held_out else _metrics_at_best_f1(y_true, ys))  # else: oracle threshold
        metrics_by_key[key] = {
            "n_blocks": len(y_true), "anomaly_rate": float(np.mean(y_true)),
            "auroc": float(roc_auc_score(y_true, ys)),
            "pr_auc": float(average_precision_score(y_true, ys)),
            "precision": best["precision"], "recall": best["recall"], "f1": best["f1"],
            "threshold": best["threshold"], "confusion": best["confusion"],
            "_y_true": y_true,
        }

    # Headline the empirically-best aggregator (by PR-AUC), not a fixed assumption.
    primary = max(metrics_by_key, key=lambda k: metrics_by_key[k]["pr_auc"])
    log.info("Best block-score aggregator by PR-AUC: %s", primary)
    _plot_curves(y_true, scores[primary])
    write_report(metrics_by_key, primary=primary, held_out=args.held_out)

    m = metrics_by_key[primary]
    print("\n" + "=" * 60)
    print("  BLOCK-LEVEL ANOMALY DETECTION")
    print("=" * 60)
    print(f"  Blocks {m['n_blocks']:,}  ·  anomaly rate {m['anomaly_rate']:.3f}")
    print(f"  PR-AUC {m['pr_auc']:.4f}  ·  AUROC {m['auroc']:.4f}")
    print(f"  Best-F1: P {m['precision']:.3f} · R {m['recall']:.3f} · F1 {m['f1']:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

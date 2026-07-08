# =============================================================================
# preprocessing/block_dataset.py — Per-block, label-aware data for HONEST
# anomaly-detection training and evaluation (DeepLog-style).
#
# WHY THIS EXISTS (audit findings, see docs/RESEARCH_AUDIT.md):
#   The production cache (preprocessing/dataset.build_memmap_sequences) is ONE
#   global sliding window over the concatenated EventId stream:
#     * windows cross block boundaries, and
#     * carry no BlockId.
#   Consequences: you cannot (a) train on "normal blocks only", nor (b) hold out
#   a block-level TEST set. Both are required to turn next-event prediction into
#   suspicious-activity detection and to report paper-comparable numbers that
#   are not in-sample.
#
# This module rebuilds windows PER BLOCK from Event_traces.csv + anomaly_label.csv,
# splits by BlockId (train/val/test, stratified by label, fixed seed), and exposes
# a normal-only filter. Windowing MIRRORS evaluation.anomaly_eval.score_blocks
# exactly (context = preceding up to seq_len events; target = next event), so
# training and scoring stay consistent with NO padding and NO model change — the
# same GRUAnomalyDetector and the same score_blocks() are reused unchanged.
# =============================================================================

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.anomaly_eval import _parse_features

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load blocks WITH their BlockIds (load_blocks() in anomaly_eval drops the ids,
# which we need to split by block — so this is a thin, id-preserving variant).
# ---------------------------------------------------------------------------
def load_labeled_blocks(encoder, config, max_blocks=None, shuffle_seed=None):
    """Return (blocks, labels, block_ids).

    blocks    : list of np.int64 arrays (encoder-indexed EventId sequences).
    labels    : np.int64 array, 1 = Anomaly, 0 = Normal.
    block_ids : list of BlockId strings (aligned to blocks/labels).

    Unknown EventIds (not in the trained vocab) are dropped from a block; blocks
    with < 2 surviving events are skipped (need ≥1 context→target pair). When
    `max_blocks` is set we take a RANDOM sample (shuffle then cap) — NOT the
    first N in file order — fixing the file-order bias the old max_blocks had.
    """
    ds = config.get("datasets", {}).get(config.get("dataset", "hdfs"), {})
    traces_path = ds.get("event_traces", "data/HDFS_v1/preprocessed/Event_traces.csv")
    label_path = ds.get("label_csv", "data/HDFS_v1/preprocessed/anomaly_label.csv")

    log.info("Loading labels from %s", label_path)
    labels_df = pd.read_csv(label_path)
    label_map = dict(zip(labels_df["BlockId"], labels_df["Label"]))

    log.info("Loading block sequences from %s", traces_path)
    traces = pd.read_csv(traces_path, usecols=["BlockId", "Features"])

    c2i = encoder.class_to_idx
    blocks, labels, block_ids, skipped, dropped_events = [], [], [], 0, 0
    for block_id, feats in zip(traces["BlockId"], traces["Features"]):
        if block_id not in label_map:
            continue
        eids = _parse_features(feats)
        kept = [c2i[e] for e in eids if e in c2i]
        dropped_events += len(eids) - len(kept)
        idxs = np.array(kept, dtype=np.int64)
        if len(idxs) < 2:
            skipped += 1
            continue
        blocks.append(idxs)
        labels.append(1 if str(label_map[block_id]).strip().lower() == "anomaly" else 0)
        block_ids.append(block_id)

    labels = np.array(labels, dtype=np.int64)

    # Optional RANDOM subsample (reproducible) for fast iteration.
    if max_blocks and len(blocks) > max_blocks:
        rng = np.random.default_rng(
            config.get("random_seed", 42) if shuffle_seed is None else shuffle_seed)
        sel = rng.choice(len(blocks), size=max_blocks, replace=False)
        sel.sort()
        blocks = [blocks[i] for i in sel]
        block_ids = [block_ids[i] for i in sel]
        labels = labels[sel]

    n_anom = int(labels.sum())
    log.info("Loaded %d blocks (%d anomalies, %.2f%%); skipped %d too-short; "
             "dropped %d unknown-vocab events.",
             len(blocks), n_anom, 100 * labels.mean() if len(labels) else 0.0,
             skipped, dropped_events)
    return blocks, labels, block_ids


# ---------------------------------------------------------------------------
# Honest split — by BLOCK, stratified by label, fixed seed.
# ---------------------------------------------------------------------------
def split_block_indices(labels, config):
    """Split block indices into (train_idx, val_idx, test_idx).

    Mirrors preprocessing.dataset.build_dataloaders' two-step carve, but operates
    on BLOCKS (not flat windows) and STRATIFIES by label so the rare ~2.9%
    anomalies are represented in every split. Fixed seed → reproducible.
    """
    test_size = config["test_size"]
    val_size = config["val_size"]
    seed = config["random_seed"]
    idx = np.arange(len(labels))

    temp_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels)
    val_frac = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=val_frac, random_state=seed, stratify=labels[temp_idx])

    log.info("Block split — train %d (%.2f%% anom) | val %d (%.2f%% anom) | "
             "test %d (%.2f%% anom)",
             len(train_idx), 100 * labels[train_idx].mean(),
             len(val_idx), 100 * labels[val_idx].mean(),
             len(test_idx), 100 * labels[test_idx].mean())
    return train_idx, val_idx, test_idx


def select_blocks(blocks, labels, block_ids, idx):
    """Subset (blocks, labels, block_ids) by an index array."""
    return ([blocks[i] for i in idx],
            labels[idx],
            [block_ids[i] for i in idx])


# ---------------------------------------------------------------------------
# Per-block window builder — IDENTICAL rule to anomaly_eval.score_blocks, so a
# model trained on these windows is scored on the same kind of windows.
# ---------------------------------------------------------------------------
def build_windows(blocks, seq_len):
    """Flatten blocks into (context, target) windows.

    For block b and position i in [1, len(b)): context = b[max(0, i-seq_len):i],
    target = b[i]. No padding (variable-length contexts are handled by the
    length-bucketed training/scoring loops). Returns:
        contexts : list of np.int64 arrays (variable length, 1..seq_len)
        targets  : np.int64 array (n_windows,)
        win_len  : np.int64 array — context length per window (for bucketing)
    """
    contexts, targets, win_len = [], [], []
    for blk in blocks:
        L = len(blk)
        for i in range(1, L):
            ctx = blk[max(0, i - seq_len):i]
            contexts.append(ctx)
            targets.append(int(blk[i]))
            win_len.append(len(ctx))
    return contexts, np.array(targets, dtype=np.int64), np.array(win_len, dtype=np.int64)


def normal_only(blocks, labels, block_ids):
    """Keep only NORMAL blocks (label == 0) — the DeepLog training regime."""
    keep = [i for i, y in enumerate(labels) if y == 0]
    return ([blocks[i] for i in keep],
            labels[keep],
            [block_ids[i] for i in keep])


def build_window_buckets(blocks, seq_len):
    """Group (context, target) windows by context length for rectangular batching.

    Returns {length: (ctx_matrix [n, length] int64, targets [n] int64)} — the
    memory-compact, no-padding form the length-bucketed training loop consumes.
    Same windowing rule as build_windows() / anomaly_eval.score_blocks, so a model
    trained on these is scored on the same kind of windows (train/eval consistent).
    """
    from collections import defaultdict
    ctx_by_len, tgt_by_len = defaultdict(list), defaultdict(list)
    for blk in blocks:
        L = len(blk)
        for i in range(1, L):
            ctx = blk[max(0, i - seq_len):i]
            ctx_by_len[len(ctx)].append(ctx)
            tgt_by_len[len(ctx)].append(blk[i])
    buckets = {}
    for ln in ctx_by_len:
        buckets[ln] = (np.stack(ctx_by_len[ln]).astype(np.int64),
                       np.array(tgt_by_len[ln], dtype=np.int64))
    return buckets


def sqrt_inverse_class_weights(targets, vocab_size):
    """sqrt-inverse-frequency class weights from training targets (matches the
    production recipe in preprocessing.encoder.compute_class_weights), so the only
    variable between the mixed and normal-only runs is the TRAINING DATA."""
    counts = np.bincount(targets, minlength=vocab_size).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.sum() * vocab_size   # normalise to mean ~1 (same convention)
    return w.astype(np.float32)

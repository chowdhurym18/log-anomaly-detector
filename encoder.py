# =============================================================================
# preprocessing/encoder.py — Streaming vocabulary builder + cache validation.
#
# Problem at scale: loading millions of rows into a Python list eats RAM fast.
# Fix: scan the CSV in chunks, accumulate only a Counter of unique EventIds.
# Also holds cache-integrity checks and sqrt-inverse-frequency class weights.
# =============================================================================

import pickle
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch

from config import CONFIG

log = logging.getLogger(__name__)


class StreamingLabelEncoder:
    """
    A minimal label encoder that streams through the CSV to build its
    vocabulary, then encodes without ever holding the full dataset in RAM.

    Why not sklearn's LabelEncoder?
        sklearn's LabelEncoder.fit() requires ALL data in memory at once.
        With 50 M rows that's ~400 MB just for the string list.
        This version processes the file one chunk at a time.
    """

    def __init__(self):
        self.classes_ = None          # Sorted list of unique EventId strings
        self.class_to_idx = None      # Dict: EventId string → integer
        self._fitted = False

    def fit(self, csv_path: str, col: str, chunksize: int):
        """
        First pass over the CSV: count every unique EventId.

        Memory cost = O(vocabulary_size) — typically only a few hundred
        unique events even in multi-million-row HDFS logs.
        """
        log.info("Building vocabulary (streaming pass 1 of 2) ...")
        counter = Counter()
        total_rows = 0
        for chunk in pd.read_csv(csv_path, usecols=[col], chunksize=chunksize):
            counter.update(chunk[col].dropna().tolist())
            total_rows += len(chunk)
            log.info(f"  scanned {total_rows:,} rows, vocab so far: {len(counter)}")

        self.classes_      = sorted(counter.keys())
        self.class_to_idx  = {c: i for i, c in enumerate(self.classes_)}
        self._fitted       = True
        log.info(f"Vocabulary size: {len(self.classes_)} unique EventIds")
        log.info(f"Total rows scanned: {total_rows:,}")
        return self

    def transform_chunk(self, series: pd.Series) -> np.ndarray:
        """Encodes a pandas Series of EventId strings to a numpy int32 array."""
        return np.array([self.class_to_idx[v] for v in series if v in self.class_to_idx],
                        dtype=np.int32)

    def inverse_transform(self, indices, context: str = "decode"):
        """Decodes integer indices back to EventId strings."""
        if not self._fitted or self.classes_ is None:
            raise ValueError("StreamingLabelEncoder must be fitted before decoding.")

        vocab_size = len(self.classes_)
        decoded = []
        sequence = list(indices)

        for pos, raw_idx in enumerate(sequence):
            idx = int(raw_idx)
            if idx < 0 or idx >= vocab_size:
                log.error(
                    "Invalid encoded EventId during %s: offending index=%s, "
                    "vocabulary size=%s, position=%s, sequence=%s",
                    context, idx, vocab_size, pos, sequence,
                )
                decoded.append(f"<UNK:{idx}>")
                continue
            decoded.append(self.classes_[idx])

        return decoded

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"classes": self.classes_, "c2i": self.class_to_idx}, f)

    @classmethod
    def load(cls, path: str):
        enc = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        enc.classes_     = d["classes"]
        enc.class_to_idx = d["c2i"]
        enc._fitted      = True
        return enc


def validate_encoded_values(name: str, values, encoder: StreamingLabelEncoder,
                            sequence=None) -> bool:
    """
    Checks that encoded EventId values are valid for the fitted vocabulary.

    Returns True when all values are in range. Logs the first offending index
    with enough context to diagnose stale caches or bad decode inputs.
    """
    vocab_size = len(encoder.classes_)
    arr = np.asarray(values)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    chunk_size = 100_000
    for start in range(0, arr.shape[0], chunk_size):
        chunk = arr[start:start + chunk_size]
        invalid = (chunk < 0) | (chunk >= vocab_size)

        if not invalid.any():
            continue

        first = np.argwhere(invalid)[0]
        first_tuple = tuple(int(i) for i in first)
        offending_idx = int(chunk[first_tuple])
        position = (start + first_tuple[0],) + first_tuple[1:]
        bad_sequence = sequence if sequence is not None else (
            chunk[first_tuple[0]].tolist() if chunk.ndim > 1 else chunk.tolist()
        )

        log.error(
            "Invalid encoded EventId in %s: offending index=%s, vocabulary size=%s, "
            "position=%s, sequence=%s",
            name, offending_idx, vocab_size, position, bad_sequence,
        )
        return False

    return True


def cache_is_valid(X_path: str, y_path: str, meta: dict,
                   encoder: StreamingLabelEncoder) -> bool:
    """
    Validates cached preprocessing before reuse.

    A stale cache can pair X/y arrays produced by an older encoder with a newer
    encoder.pkl. Training can still finish, but later sequence decoding may try
    to index past encoder.classes_ and crash.
    """
    if meta.get("vocab_size") != len(encoder.classes_):
        log.warning(
            "Cached vocab size mismatch: meta=%s, encoder=%s",
            meta.get("vocab_size"), len(encoder.classes_),
        )
        return False

    if "seq_len" in meta and meta["seq_len"] != CONFIG["sequence_length"]:
        log.warning(
            "Cached seq_len mismatch: meta=%s, config=%s — rebuilding sequences",
            meta["seq_len"], CONFIG["sequence_length"],
        )
        return False

    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    if X.shape[0] != meta.get("n_sequences") or y.shape[0] != meta.get("n_sequences"):
        log.warning(
            "Cached sequence count mismatch: meta=%s, X=%s, y=%s",
            meta.get("n_sequences"), X.shape[0], y.shape[0],
        )
        return False

    return (
        validate_encoded_values("cached X sequences", X, encoder) and
        validate_encoded_values("cached y labels", y, encoder)
    )


def log_distribution(name: str, values, encoder: StreamingLabelEncoder, top_k: int = 10):
    """Logs decoded EventId counts for labels or predictions."""
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=len(encoder.classes_))
    total = int(counts.sum())
    log.info("%s distribution (total=%s, vocab=%s):", name, total, len(encoder.classes_))

    for idx in counts.argsort()[::-1][:top_k]:
        if counts[idx] == 0:
            continue
        event_id = encoder.inverse_transform([int(idx)], context=f"{name} distribution")[0]
        log.info("  %s (%s): %s  %.1f%%", event_id, idx, int(counts[idx]), 100 * counts[idx] / total)


def log_preprocessing_debug(X_path: str, y_path: str, train_idx: np.ndarray,
                            test_idx: np.ndarray, encoder: StreamingLabelEncoder):
    """Logs vocabulary and EventId frequency diagnostics for cached sequences."""
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    log.info("Vocabulary size: %s", len(encoder.classes_))
    log.info("Vocabulary mapping: %s", {event: i for i, event in enumerate(encoder.classes_)})
    log.info("Encoded sequence range: X=[%s, %s], y=[%s, %s]",
             int(X.min()), int(X.max()), int(y.min()), int(y.max()))

    log_distribution("All target labels", y, encoder)
    log_distribution("Train target labels", y[train_idx], encoder)
    log_distribution("Test target labels", y[test_idx], encoder)


def verify_sequence_alignment(csv_path: str, X_path: str, y_path: str,
                              encoder: StreamingLabelEncoder, seq_len: int,
                              chunksize: int, num_checks: int = 5):
    """
    Verifies sliding-window alignment:
    X[i] must equal encoded_events[i:i+seq_len], y[i] must equal encoded_events[i+seq_len].
    """
    encoded_parts = []
    needed = seq_len + num_checks

    for chunk in pd.read_csv(csv_path, usecols=["EventId"], chunksize=chunksize):
        encoded_parts.append(encoder.transform_chunk(chunk["EventId"].dropna()))
        if sum(len(part) for part in encoded_parts) >= needed:
            break

    encoded = np.concatenate(encoded_parts)[:needed]
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    for i in range(min(num_checks, len(y))):
        expected_x = encoded[i:i + seq_len]
        expected_y = encoded[i + seq_len]
        if not np.array_equal(X[i], expected_x) or int(y[i]) != int(expected_y):
            log.error(
                "Sliding-window alignment mismatch at row %s: X=%s expected_X=%s y=%s expected_y=%s",
                i, X[i].tolist(), expected_x.tolist(), int(y[i]), int(expected_y),
            )
            return False

    log.info("Sliding-window alignment verified for first %s sequences.", min(num_checks, len(y)))
    return True


def compute_class_weights(y_path: str, train_idx: np.ndarray, vocab_size: int,
                          device: torch.device):
    """
    Sqrt-inverse-frequency class weights.

    weight[c] = sqrt(N / (V * count[c]))  — normalised so the mean over
    present classes equals 1.

    WHY this instead of effective-number-of-samples (beta=0.999):
        With HDFS_2k, E12 has 1 training sample and E6 has ~219.
        Effective-number gives E12 weight=8.06, E6 weight=0.04.
        Their total gradient signals (count × weight) are 8 vs 9 — EQUAL.
        The model collapses to always predicting E12 (the rarest class),
        because minimising loss on 1 heavily-weighted sample is easier than
        219 lightly-weighted ones.  Result: 0.25% accuracy.

        Sqrt-inverse gives E12 weight=4.5, E6 weight=0.3.
        Total signals: E12=4.5 vs E6=66 — E6 dominates 15×, which is
        correct.  This is the approach that previously achieved 38%.
    """
    y = np.load(y_path, mmap_mode="r")
    n_train = len(train_idx)
    counts = np.bincount(y[train_idx].astype(np.int64), minlength=vocab_size)
    safe = np.maximum(counts, 1)
    weights = np.sqrt(n_train / (vocab_size * safe))
    weights[counts == 0] = 0.0
    weights = weights / weights.mean()
    log.info("Class weights (sqrt-inverse-freq): %s", np.round(weights, 3).tolist())
    present = weights[counts > 0]
    log.info("  min=%.3f  max=%.3f  ratio=%.1fx", present.min(), present.max(),
             present.max() / max(present.min(), 1e-9))
    return torch.tensor(weights, dtype=torch.float32, device=device)

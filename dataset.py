# =============================================================================
# preprocessing/dataset.py — Memory-mapped sequence arrays + DataLoaders.
#
# Problem at scale: np.array(X) with 10 M sequences × 20 steps = 200 M int32
#   in RAM. Fine for 2k logs, out-of-memory for millions.
# Fix: write sequences directly to disk via np.memmap so the OS pages them
#   in and out as needed. The DataLoader reads slices, not the whole array.
# =============================================================================

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocessing.encoder import StreamingLabelEncoder

log = logging.getLogger(__name__)


def build_memmap_sequences(csv_path: str, encoder: StreamingLabelEncoder,
                           seq_len: int, chunksize: int, cache_dir: str):
    """
    Second pass over the CSV: encodes events and writes overlapping windows
    directly into memory-mapped numpy files on disk.

    Memory cost during creation = O(chunksize) — the overlap buffer between
    consecutive chunks is the only extra allocation.

    Args:
        csv_path  : Path to the structured log CSV.
        encoder   : Fitted StreamingLabelEncoder.
        seq_len   : Sliding window size.
        chunksize : Rows per chunk.
        cache_dir : Directory to store the .npy memmap files.

    Returns:
        X_path, y_path : Paths to the memmap files.
        n_sequences    : Total number of (window, label) pairs created.
    """
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    X_path = str(cache / "X.npy")
    y_path = str(cache / "y.npy")

    # ---- Count total sequences first (needed to pre-allocate memmap) ----
    log.info("Counting total rows to pre-allocate memory-mapped arrays ...")
    total_rows = 0
    total_events = 0
    for chunk in pd.read_csv(csv_path, usecols=["EventId"], chunksize=chunksize):
        total_rows += len(chunk)
        total_events += len(encoder.transform_chunk(chunk["EventId"].dropna()))

    skipped_rows = total_rows - total_events
    if skipped_rows:
        log.warning(
            "Skipped %s rows with missing or unknown EventId values while building sequences.",
            skipped_rows,
        )

    # Each row except the last seq_len rows produces one (window, label) pair
    n_sequences = max(0, total_events - seq_len)
    log.info(f"  Total encoded events: {total_events:,} → {n_sequences:,} sequences")

    # ---- Pre-allocate on disk ----
    # dtype=int32 halves memory vs int64; EventId vocab is < 2^31
    X_mmap = np.lib.format.open_memmap(X_path, mode="w+",
                                        dtype=np.int32,
                                        shape=(n_sequences, seq_len))
    y_mmap = np.lib.format.open_memmap(y_path, mode="w+",
                                        dtype=np.int32,
                                        shape=(n_sequences,))

    # ---- Stream, encode, and fill ----
    log.info("Encoding and writing sequences (streaming pass 2 of 2) ...")
    carry   = np.array([], dtype=np.int32)  # leftover events from previous chunk
    written = 0                              # index into the memmap arrays

    for chunk in pd.read_csv(csv_path, usecols=["EventId"], chunksize=chunksize):
        encoded_chunk = encoder.transform_chunk(chunk["EventId"].dropna())
        # Prepend the carry-over buffer so windows straddle chunk boundaries
        events = np.concatenate([carry, encoded_chunk])

        # How many full windows fit in this buffer?
        n_windows = len(events) - seq_len
        if n_windows > 0:
            # Vectorised window creation using stride tricks — no Python loop
            # shape: (n_windows, seq_len+1) where last col is the label
            strides = np.lib.stride_tricks.sliding_window_view(events, seq_len + 1)
            end = min(written + n_windows, n_sequences)
            actual = end - written
            X_mmap[written:end] = strides[:actual, :seq_len]
            y_mmap[written:end] = strides[:actual,  seq_len]
            written += actual

        # Keep the last seq_len events as the carry buffer for next chunk
        carry = events[-seq_len:]
        log.info(f"  written {written:,} / {n_sequences:,} sequences")

    # Flush to disk
    X_mmap.flush()
    y_mmap.flush()
    del X_mmap, y_mmap
    gc.collect()

    log.info(f"Sequence arrays saved to: {cache_dir}/")
    return X_path, y_path, n_sequences


class MemmapLogDataset(Dataset):
    """
    A PyTorch Dataset backed by memory-mapped numpy arrays on disk.

    __getitem__ reads only the requested row from disk (or OS cache),
    so memory usage is constant regardless of dataset size.

    Args:
        X_path   : Path to the memmap X array.
        y_path   : Path to the memmap y array.
        indices  : Subset of row indices (for train/test split without copying).
    """

    def __init__(self, X_path: str, y_path: str, indices: np.ndarray):
        self.X       = np.load(X_path, mmap_mode="r")  # read-only memmap
        self.y       = np.load(y_path, mmap_mode="r")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        # Copy the slice to a writeable array before converting to tensor
        x = torch.tensor(self.X[idx].copy(), dtype=torch.long)
        y = torch.tensor(int(self.y[idx]),   dtype=torch.long)
        return x, y


def build_dataloaders(X_path, y_path, n_sequences, test_size, val_size,
                      random_seed, batch_size, num_workers, pin_memory,
                      prefetch_factor,
                      max_train=None, max_val=None, max_test=None):
    """
    Creates train / validation / test DataLoaders from memmap arrays.

    Splitting strategy:
        1. Carve out test_size fraction as held-out test set.
        2. From the remainder, carve out val_size/(1-test_size) fraction
           as a validation set used for early stopping and LR scheduling.
        3. Everything left is the training set.

    The split is done on indices only — no data is duplicated or moved.

    Optional subset limits (max_train / max_val / max_test) shrink each split to
    at most N sequences. This powers the --experiment / --max-* flags for fast,
    lightweight runs. Because train_test_split has already shuffled the indices
    (with a fixed random_state), taking a head slice yields a *reproducible random
    subset* — not a biased "first N rows". Passing None (the default) keeps the
    full split, so existing behaviour is unchanged. These limits only shrink the
    in-memory index arrays; the on-disk memmap cache (.gru_cache/) is untouched
    and stays valid.

    Returns:
        train_loader, val_loader, test_loader : PyTorch DataLoaders.
        train_idx, val_idx, test_idx          : The split index arrays.
    """
    log.info("Splitting indices into train / val / test sets ...")
    all_idx = np.arange(n_sequences)

    # Step 1: hold out test set
    temp_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_seed
    )
    # Step 2: carve validation from the training pool
    val_fraction_of_temp = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=val_fraction_of_temp, random_state=random_seed
    )

    # Step 3 (optional): cap each split for lightweight / experiment runs.
    if max_train is not None:
        train_idx = train_idx[:max_train]
    if max_val is not None:
        val_idx = val_idx[:max_val]
    if max_test is not None:
        test_idx = test_idx[:max_test]
    if any(m is not None for m in (max_train, max_val, max_test)):
        log.info("  Subset limits applied (max_train=%s, max_val=%s, max_test=%s).",
                 max_train, max_val, max_test)

    log.info(
        "  Train: %s  |  Val: %s  |  Test: %s",
        f"{len(train_idx):,}", f"{len(val_idx):,}", f"{len(test_idx):,}",
    )

    train_ds = MemmapLogDataset(X_path, y_path, train_idx)
    val_ds   = MemmapLogDataset(X_path, y_path, val_idx)
    test_ds  = MemmapLogDataset(X_path, y_path, test_idx)

    loader_kwargs = dict(
        batch_size         = batch_size,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        prefetch_factor    = prefetch_factor if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx

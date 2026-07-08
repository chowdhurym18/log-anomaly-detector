# =============================================================================
# compare_models.py — Head-to-head comparison of the three model variants.
#
# Trains (or reuses) each of: gru, bigru, bigru_attention on the SAME data split,
# evaluates each with the existing evaluate_model(), and prints a publication-style
# table (Top-1, Top-3, Weighted F1, Macro F1, training time, inference time). Also
# writes outputs/model_comparison.csv and outputs/model_comparison.md.
#
# Reuses the existing pipeline pieces — no logic is duplicated:
#   main._load_or_build_cache, build_dataloaders, compute_class_weights,
#   train_model, evaluate_model, config.variant_flags.
#
# Usage:
#   venv/bin/python compare_models.py                       # all variants, full epochs
#   venv/bin/python compare_models.py --reuse-existing      # reuse saved checkpoints
#   venv/bin/python compare_models.py --variants bigru_attention --epochs 1 --limit 50000
# =============================================================================

import os
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CONFIG, get_device, variant_flags
from utils.logging_utils import setup_logging
from preprocessing.dataset import build_dataloaders
from preprocessing.encoder import compute_class_weights
from models.gru_model import GRUAnomalyDetector
from training.train import train_model
from evaluation.evaluate import evaluate_model
from main import _load_or_build_cache, checkpoint_dir_for

log = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="Compare GRU / BiGRU / BiGRU+Attention.")
    p.add_argument("--variants", nargs="+",
                   default=["gru", "bigru", "bigru_attention"],
                   choices=["gru", "bigru", "bigru_attention"],
                   help="Variants to compare (default: all three).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override CONFIG['num_epochs'] (use 1 for a quick smoke run).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Use only the first N sequences (quick/smoke runs).")
    p.add_argument("--reuse-existing", action="store_true",
                   help="Load a variant's saved best_model.pt if present instead of retraining.")
    return p.parse_args()


def _build_model(variant, vocab_size, device):
    flags = variant_flags(variant)
    return GRUAnomalyDetector(
        vocab_size    = vocab_size,
        embedding_dim = CONFIG["embedding_dim"],
        hidden_dim    = CONFIG["hidden_dim"],
        num_layers    = CONFIG["num_layers"],
        dropout       = CONFIG["dropout"],
        bidirectional = flags["bidirectional"],
        use_attention = flags["use_attention"],
    ).to(device)


def _sync(device):
    """Block until the device finishes queued work (so timings are accurate)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _inference_ms_per_1k(model, test_loader, device, n_batches: int = 50) -> float:
    """Pure forward-pass speed: time ONLY model(x), averaged over up to n_batches,
    reported as milliseconds per 1,000 sequences.

    This deliberately excludes the sklearn metric computation and array
    concatenation that evaluate_model does — those are eval bookkeeping, not
    inference. The host→device transfer happens before the timer; the device is
    synchronised around the forward call so async kernels are fully counted.
    """
    model.eval()
    total_seqs, elapsed = 0, 0.0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= n_batches:
                break
            x = x.to(device, non_blocking=True)
            _sync(device)
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            elapsed   += time.perf_counter() - t0
            total_seqs += x.shape[0]
    if total_seqs == 0:
        return float("nan")
    return (elapsed / total_seqs) * 1000 * 1000   # ms per 1,000 sequences


def _count_epochs(ckpt_dir: str) -> int:
    """How many epoch_NNN.pt files a variant produced (honest training-time context)."""
    return len([f for f in os.listdir(ckpt_dir) if f.startswith("epoch_")]) \
        if os.path.isdir(ckpt_dir) else 0


def _evaluate_one(variant, loaders, encoder, vocab_size, class_weights, device, args):
    """Build → (load or train) → evaluate one variant. Returns a metrics row."""
    train_loader, val_loader, test_loader, train_idx, _, test_idx = loaders
    # Re-seed right before building so every variant starts from the SAME
    # initialisation RNG state (fair, reproducible comparison regardless of order).
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    model    = _build_model(variant, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_dir  = checkpoint_dir_for(variant)
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    # ---- Load existing checkpoint, or train ----
    train_time = None
    if args.reuse_existing and os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        log.info("[%s] reused checkpoint %s", variant, best_ckpt)
    else:
        log.info("[%s] training (%d params) ...", variant, n_params)
        t0 = time.time()
        model = train_model(
            model, train_loader, val_loader,
            num_epochs              = args.epochs or CONFIG["num_epochs"],
            learning_rate           = CONFIG["learning_rate"],
            weight_decay            = CONFIG.get("weight_decay", 1e-4),
            accum_steps             = CONFIG["accum_steps"],
            lr_patience             = CONFIG["lr_patience"],
            early_stopping_patience = CONFIG.get("early_stopping_patience", 10),
            checkpoint_dir          = ckpt_dir,
            device                  = device,
            class_weights           = class_weights,
            label_smoothing         = CONFIG.get("label_smoothing", 0.0),
        )
        train_time = time.time() - t0

    # ---- Evaluate quality (NOT timed) and inference speed (pure forward pass) ----
    metrics  = evaluate_model(model, test_loader, device, encoder=encoder)
    infer_ms = _inference_ms_per_1k(model, test_loader, device)

    return {
        "variant":        variant,
        "params":         n_params,
        "epochs":         _count_epochs(ckpt_dir),
        "top1":           metrics["top1_accuracy"],
        "top3":           metrics["top3_accuracy"],
        "weighted_f1":    metrics["weighted_f1"],
        "macro_f1":       metrics["macro_f1"],
        "train_time_s":   train_time,
        "infer_ms_per_1k": infer_ms,
    }


def _print_table(rows):
    hdr = (f"{'Variant':<17} {'Params':>9} {'Ep':>3} {'Top-1':>8} {'Top-3':>8} "
           f"{'wF1':>8} {'macroF1':>8} {'Train(s)':>10} {'Inf(ms/1k)':>11}")
    print("\n" + "=" * len(hdr))
    print("  MODEL COMPARISON  (inference = pure forward pass; train time over epochs reached)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tt = f"{r['train_time_s']:.1f}" if r["train_time_s"] is not None else "reused"
        print(f"{r['variant']:<17} {r['params']:>9,} {r['epochs']:>3} "
              f"{r['top1']*100:>7.2f}% {r['top3']*100:>7.2f}% "
              f"{r['weighted_f1']:>8.4f} {r['macro_f1']:>8.4f} "
              f"{tt:>10} {r['infer_ms_per_1k']:>11.2f}")
    print("=" * len(hdr) + "\n")


def _write_outputs(rows):
    Path("outputs").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/model_comparison.csv", index=False)

    # Markdown table for the poster / README.
    lines = ["| Variant | Params | Epochs | Top-1 | Top-3 | Weighted F1 | Macro F1 | Train (s) | Inf (ms/1k) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tt = f"{r['train_time_s']:.1f}" if r["train_time_s"] is not None else "reused"
        lines.append(
            f"| {r['variant']} | {r['params']:,} | {r['epochs']} | {r['top1']*100:.2f}% | "
            f"{r['top3']*100:.2f}% | {r['weighted_f1']:.4f} | {r['macro_f1']:.4f} | "
            f"{tt} | {r['infer_ms_per_1k']:.2f} |")
    Path("outputs/model_comparison.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote outputs/model_comparison.csv and outputs/model_comparison.md")


def main():
    setup_logging()
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    args   = _parse_args()
    device = get_device()
    log.info("Comparison device: %s | variants: %s", device, args.variants)

    # ---- Shared data (built once, identical split for every variant) ----
    encoder, n_sequences, vocab_size, X_path, y_path = _load_or_build_cache(eval_only=False)
    if args.limit:
        n_sequences = min(args.limit, n_sequences)
        log.info("Using a subset of the first %s sequences (--limit).", f"{n_sequences:,}")

    loaders = build_dataloaders(
        X_path, y_path, n_sequences,
        test_size       = CONFIG["test_size"],
        val_size        = CONFIG["val_size"],
        random_seed     = CONFIG["random_seed"],
        batch_size      = CONFIG["batch_size"],
        num_workers     = CONFIG["num_workers"],
        pin_memory      = CONFIG["pin_memory"] and (device.type != "cpu"),
        prefetch_factor = CONFIG["prefetch_factor"],
    )
    train_idx = loaders[3]
    class_weights = (compute_class_weights(y_path, train_idx, vocab_size, device)
                     if CONFIG["use_class_weights"] else None)

    # ---- Run each variant ----
    rows = [_evaluate_one(v, loaders, encoder, vocab_size, class_weights, device, args)
            for v in args.variants]

    _print_table(rows)
    _write_outputs(rows)
    log.info("Comparison complete. ✓")


if __name__ == "__main__":
    main()

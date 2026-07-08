# =============================================================================
# run_normal_only.py — Task 2: Mixed vs Normal-Only training (the #1 experiment).
#
# Trains TWO fresh GRUs on the SAME honest block-level split:
#   * mixed       — all training blocks (the current regime, but on a held-out split)
#   * normal_only — NORMAL training blocks only (the DeepLog regime)
# Both are scored on the SAME held-out TEST blocks (never in-sample) with the
# existing evaluation.anomaly_eval.score_blocks. The F1 threshold is chosen on the
# VALIDATION split and applied to TEST (no oracle-threshold optimism). Everything
# else — architecture, loss recipe, seed, windows — is identical, so the ONLY
# variable is the training data.
#
# Output: outputs/normal_only_training_report.md (Detection P/R/F1, AUROC, PR-AUC).
#
# Usage (smoke):  venv/bin/python run_normal_only.py --max-blocks 30000 --epochs 4
# Usage (full):   venv/bin/python run_normal_only.py
# =============================================================================

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import CONFIG, get_device, variant_flags, config_for_dataset
from utils.logging_utils import setup_logging
from utils.perf import Timer, peak_rss_mb, device_sync
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from evaluation.anomaly_eval import score_blocks, _metrics_at_best_f1

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
)

log = logging.getLogger(__name__)

AGGREGATORS = ("max_surprisal", "mean_surprisal", "topk_miss_frac")


# ---------------------------------------------------------------------------
# Training — length-bucketed, no padding, consistent with score_blocks.
# ---------------------------------------------------------------------------
def _val_loss(model, val_buckets, device, batch):
    crit = nn.CrossEntropyLoss()
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for ln, (ctx, tgt) in val_buckets.items():
            for s in range(0, len(tgt), batch):
                bx = torch.from_numpy(ctx[s:s + batch]).long().to(device)
                by = torch.from_numpy(tgt[s:s + batch]).long().to(device)
                total += crit(model(bx), by).item()
                nb += 1
    return total / max(nb, 1)


def train_on_buckets(model, buckets, val_buckets, device, config, epochs,
                     ckpt_path, class_weights=None, patience=4, stats=None):
    """`stats`, when given a dict, receives {'epochs_run', 'best_val_loss'} —
    an out-param so existing callers (run_generalization.py) are unaffected."""
    rng = np.random.default_rng(config["random_seed"])
    batch = config["batch_size"]
    w = None if class_weights is None else torch.tensor(class_weights, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=config["lr_patience"], min_lr=1e-6)

    # Static batch plan (row indices per bucket); rows are reshuffled each epoch.
    lengths = sorted(buckets)
    best, no_improve = float("inf"), 0
    epochs_run = 0
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        batches = []
        for ln in lengths:
            ctx, tgt = buckets[ln]
            order = rng.permutation(len(tgt))
            for s in range(0, len(tgt), batch):
                batches.append((ln, order[s:s + batch]))
        rng.shuffle(batches)

        total, nb = 0.0, 0
        for ln, rows in batches:
            ctx, tgt = buckets[ln]
            bx = torch.from_numpy(ctx[rows]).long().to(device)
            by = torch.from_numpy(tgt[rows]).long().to(device)
            loss = crit(model(bx), by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1

        vloss = _val_loss(model, val_buckets, device, batch)
        sched.step(vloss)
        log.info("  epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                 epoch, epochs, total / max(nb, 1), vloss, opt.param_groups[0]["lr"])
        if vloss < best:
            best, no_improve = vloss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  early stop (no val improvement for %d epochs)", patience)
                break

    if stats is not None:
        stats["epochs_run"] = epochs_run
        stats["best_val_loss"] = best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# Evaluation — threshold chosen on VAL, applied to TEST (honest operating point).
# ---------------------------------------------------------------------------
def evaluate(model, val_blocks, val_y, test_blocks, test_y, device, config):
    val_scores = score_blocks(model, val_blocks, device, config)
    test_scores = score_blocks(model, test_blocks, device, config)
    out = {}
    for key in AGGREGATORS:
        thr = _metrics_at_best_f1(val_y, val_scores[key])["threshold"]  # picked on VAL
        y_pred = (test_scores[key] >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            test_y, y_pred, average="binary", zero_division=0)
        out[key] = {
            "auroc": float(roc_auc_score(test_y, test_scores[key])),
            "pr_auc": float(average_precision_score(test_y, test_scores[key])),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "threshold": float(thr),
            "val_pr_auc": float(average_precision_score(val_y, val_scores[key])),
            "oracle_f1": _metrics_at_best_f1(test_y, test_scores[key])["f1"],
        }
    return out


def _fresh_model(encoder, device, config=CONFIG):
    flags = variant_flags("gru")
    return GRUAnomalyDetector(
        len(encoder.classes_), config["embedding_dim"], config["hidden_dim"],
        config["num_layers"], config["dropout"], **flags,
    ).to(device)


def _headline(metrics):
    """Aggregator with the best VALIDATION PR-AUC (chosen on val, not test)."""
    return max(AGGREGATORS, key=lambda k: metrics[k]["val_pr_auc"])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(results, meta, costs=None, out_path="outputs/normal_only_training_report.md",
                 dataset="hdfs"):
    regimes = [r for r in ("mixed", "normal_only") if r in results]
    L = []; a = L.append
    a("# Normal-Only vs Mixed Training — Block-Level Anomaly Detection\n")
    a("_Task 2 (dataset: %s). Both models are FRESH GRUs trained on the SAME honest "
      "BlockId split (stratified, seed %d); identical architecture/loss/seed, the ONLY "
      "difference is the training data. Scored on the SAME held-out TEST blocks (never "
      "in-sample). The F1 threshold is selected on VALIDATION and applied to TEST._\n"
      % (dataset, CONFIG["random_seed"]))
    a(f"- Blocks: train **{meta['n_train']:,}** · val **{meta['n_val']:,}** · "
      f"test **{meta['n_test']:,}**  (test anomaly base rate **{meta['test_rate']:.3f}**)")
    a(f"- Normal-only training blocks: **{meta['n_train_normal']:,}** "
      f"(mixed uses all {meta['n_train']:,})\n")

    a("## Headline (best aggregator by validation PR-AUC)\n")
    a("| Regime | Aggregator | Precision | Recall | F1 | AUROC | PR-AUC |")
    a("|---|---|---|---|---|---|---|")
    for regime in regimes:
        m = results[regime]
        h = _headline(m)
        x = m[h]
        a(f"| {regime} | {h} | {x['precision']:.4f} | {x['recall']:.4f} | "
          f"**{x['f1']:.4f}** | {x['auroc']:.4f} | **{x['pr_auc']:.4f}** |")
    a("")

    if len(regimes) == 2:
        mh, nh = _headline(results["mixed"]), _headline(results["normal_only"])
        dm, dn = results["mixed"][mh], results["normal_only"][nh]
        a("**Delta (normal_only − mixed):** "
          f"F1 {dn['f1'] - dm['f1']:+.4f} · AUROC {dn['auroc'] - dm['auroc']:+.4f} · "
          f"PR-AUC {dn['pr_auc'] - dm['pr_auc']:+.4f}\n")

    a("## All aggregators (TEST; threshold chosen on VAL)\n")
    a("| Regime | Aggregator | Precision | Recall | F1 | AUROC | PR-AUC | oracle-F1 |")
    a("|---|---|---|---|---|---|---|---|")
    for regime in regimes:
        for key in AGGREGATORS:
            x = results[regime][key]
            a(f"| {regime} | {key} | {x['precision']:.4f} | {x['recall']:.4f} | "
              f"{x['f1']:.4f} | {x['auroc']:.4f} | {x['pr_auc']:.4f} | {x['oracle_f1']:.4f} |")
    a("")
    a("_oracle-F1 = best-F1 threshold picked on TEST itself (upper bound; shown only "
      "for reference — the headline F1 uses the VAL-selected threshold)._\n")

    if costs:
        a("## Measured cost (this run, wall-clock)\n")
        a("| Regime | Training (s) | Epochs run | Scoring val+test (s) | Blocks/s | "
          "Peak RSS (MiB) |")
        a("|---|---|---|---|---|---|")
        for regime in regimes:
            c = costs.get(regime)
            if not c:
                continue
            a(f"| {regime} | {c['train_s']:.1f} | {c['epochs_run']} | {c['score_s']:.1f} | "
              f"{c['blocks_per_s']:.0f} | {c['peak_rss_mb']:.0f} |")
        a("")
        a(f"_Device: {costs.get('device', '?')}. Peak RSS is process-wide and cumulative "
          "(a later regime includes memory high-water marks of earlier ones). Scoring "
          "throughput is device-synchronized over the full val+test scoring pass._\n")

    a("## Interpretation\n")
    a("DeepLog-style detection assumes the model sees NORMAL behaviour only, so "
      "anomalous transitions remain low-probability (surprising). Training on mixed "
      "data teaches the model that anomalous transitions are ordinary, so it is not "
      "surprised by them — the diagnosed root cause. A positive normal-only delta "
      "above is direct evidence for that diagnosis.\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L) + "\n")
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Mixed vs Normal-Only training (Task 2).")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Random block subsample for fast iteration (default: all).")
    ap.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name from CONFIG['datasets'] (default: CONFIG['dataset'] = 'hdfs').")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Override the checkpoint base dir (e.g. a throwaway dir for a "
                         "timing-only run that must not touch production checkpoints).")
    ap.add_argument("--regimes", choices=["both", "mixed", "normal_only"], default="both",
                    help="Which training regime(s) to run (default: both).")
    args = ap.parse_args()

    cfg = config_for_dataset(args.dataset)
    torch.manual_seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])
    device = get_device()
    log.info("Device: %s  |  Dataset: %s", device, cfg["dataset"])

    encoder = StreamingLabelEncoder.load(os.path.join(cfg["cache_dir"], "encoder.pkl"))
    vocab = len(encoder.classes_)

    # ---- Honest block split ----
    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, cfg, max_blocks=args.max_blocks)
    tr_idx, va_idx, te_idx = bd.split_block_indices(labels, cfg)
    tr_blocks, tr_y, _ = bd.select_blocks(blocks, labels, block_ids, tr_idx)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, _ = bd.select_blocks(blocks, labels, block_ids, te_idx)

    seq_len = cfg["sequence_length"]
    val_buckets = bd.build_window_buckets(va_blocks, seq_len)   # mixed val (per spec)

    norm_blocks, _, _ = bd.normal_only(tr_blocks, tr_y, block_ids=[None] * len(tr_y))
    meta = {"n_train": len(tr_blocks), "n_val": len(va_blocks), "n_test": len(te_blocks),
            "n_train_normal": len(norm_blocks), "test_rate": float(np.mean(te_y))}

    ckpt_base = args.checkpoint_dir or cfg["block_checkpoint_dir"]
    regime_plan = [("mixed", tr_blocks), ("normal_only", norm_blocks)]
    if args.regimes != "both":
        regime_plan = [(r, b) for r, b in regime_plan if r == args.regimes]

    results, costs = {}, {"device": str(device)}
    for regime, train_blocks in regime_plan:
        log.info("=" * 60)
        log.info("  TRAINING regime=%s  (%d blocks)", regime, len(train_blocks))
        log.info("=" * 60)
        buckets = bd.build_window_buckets(train_blocks, seq_len)
        all_tgt = np.concatenate([t for _, t in buckets.values()])
        cw = bd.sqrt_inverse_class_weights(all_tgt, vocab) if cfg["use_class_weights"] else None
        model = _fresh_model(encoder, device, cfg)
        ckpt = os.path.join(ckpt_base, regime, "best_model.pt")
        tstats = {}
        with Timer() as t_train:
            model = train_on_buckets(model, buckets, val_buckets, device, cfg,
                                     epochs=args.epochs, ckpt_path=ckpt,
                                     class_weights=cw, patience=args.patience, stats=tstats)
            device_sync(device)
        with Timer() as t_score:
            results[regime] = evaluate(model, va_blocks, va_y, te_blocks, te_y, device, cfg)
            device_sync(device)
        n_scored = len(va_blocks) + len(te_blocks)
        costs[regime] = {
            "train_s": t_train.seconds,
            "epochs_run": tstats.get("epochs_run", "?"),
            "score_s": t_score.seconds,
            "blocks_per_s": n_scored / t_score.seconds if t_score.seconds > 0 else 0.0,
            "peak_rss_mb": peak_rss_mb(),
        }
        h = _headline(results[regime])
        x = results[regime][h]
        log.info("  [%s] headline=%s  F1=%.4f  AUROC=%.4f  PR-AUC=%.4f  "
                 "(train %.1fs / %s epochs, score %.1fs, peak RSS %.0f MiB)",
                 regime, h, x["f1"], x["auroc"], x["pr_auc"],
                 t_train.seconds, tstats.get("epochs_run", "?"), t_score.seconds,
                 costs[regime]["peak_rss_mb"])

    if args.checkpoint_dir:
        # Throwaway run (e.g. timing-only): keep its report next to its checkpoints
        # so the real per-dataset reports in outputs/ are never overwritten.
        out_path = os.path.join(args.checkpoint_dir, "normal_only_training_report.md")
    else:
        out_path = (os.path.join(cfg["output_dir"], "normal_only_training_report.md")
                    if cfg.get("output_dir") else "outputs/normal_only_training_report.md")
    write_report(results, meta, costs=costs, out_path=out_path, dataset=cfg["dataset"])
    print("\n" + "=" * 64)
    print("  MIXED vs NORMAL-ONLY  (held-out test; val-selected threshold)")
    print("=" * 64)
    for regime, _ in regime_plan:
        h = _headline(results[regime]); x = results[regime][h]
        print(f"  {regime:12s} [{h:14s}]  F1 {x['f1']:.4f}  AUROC {x['auroc']:.4f}  "
              f"PR-AUC {x['pr_auc']:.4f}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()

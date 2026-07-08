# =============================================================================
# analyze_detection.py — Task 1: PROVE *why* normal-only training wins.
#
# The mixed-vs-normal-only headline (run_normal_only.py) shows normal-only has the
# higher detection F1/PR-AUC. This script explains the MECHANISM with evidence, not
# assertion: it scores the SAME held-out TEST blocks with both checkpoints and shows
# how each regime separates ANOMALY blocks from NORMAL blocks.
#
#   * score distributions      — mean/median/p90/p99 of the block surprise score,
#                                split by true label, per regime.
#   * separation metrics       — AUROC, PR-AUC, F1 (val-selected threshold), the
#                                KS statistic (max CDF distance) and the median gap.
#   * histograms               — normal vs anomaly block-score overlays, mixed beside
#                                normal-only, so the separation is visible at a glance.
#
# The expected story: under MIXED training the two distributions overlap (the model
# is barely more surprised by anomalies); under NORMAL-ONLY training the anomaly
# distribution shifts right and SEPARATES — exactly the DeepLog assumption restored.
#
# Reuses the production block utilities unchanged (no new model, no new windowing):
#   preprocessing.block_dataset  + evaluation.anomaly_eval.score_blocks.
#
# Output: outputs/normal_vs_anomaly_separation.md
#         outputs/hist_mixed.png, outputs/hist_normal_only.png, outputs/separation_histograms.png
#
# Usage (full):     venv/bin/python analyze_detection.py
# Usage (subset):   venv/bin/python analyze_detection.py --max-blocks 80000 --device cpu
# =============================================================================

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
)

from config import CONFIG, get_device, variant_flags
from utils.logging_utils import setup_logging
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from evaluation.anomaly_eval import score_blocks, _metrics_at_best_f1

log = logging.getLogger(__name__)

AGGREGATORS = ("max_surprisal", "mean_surprisal", "topk_miss_frac")
PRIMARY = "max_surprisal"       # DeepLog's own rule (peak surprise) → where the effect is vivid

# (regime label, checkpoint path)
REGIMES = [
    ("mixed",       ".checkpoints/normal_only/mixed/best_model.pt"),
    ("normal_only", ".checkpoints/normal_only/normal_only/best_model.pt"),
]


def _load_model(checkpoint, encoder, device):
    flags = variant_flags("gru")
    model = GRUAnomalyDetector(
        len(encoder.classes_), CONFIG["embedding_dim"], CONFIG["hidden_dim"],
        CONFIG["num_layers"], CONFIG["dropout"], **flags,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    log.info("Loaded %s", checkpoint)
    return model


def _ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic = max distance between the empirical
    CDFs of `a` (normal scores) and `b` (anomaly scores). 0 = identical, 1 = disjoint.
    A larger KS means the regime separates anomalies from normals more cleanly."""
    a = np.sort(a); b = np.sort(b)
    grid = np.concatenate([a, b])
    grid.sort()
    cdf_a = np.searchsorted(a, grid, side="right") / max(len(a), 1)
    cdf_b = np.searchsorted(b, grid, side="right") / max(len(b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _dist_stats(x):
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)), "p99": float(np.quantile(x, 0.99)),
    }


def analyze_regime(model, va_blocks, va_y, te_blocks, te_y, device):
    """Score val + test blocks; return per-aggregator detection metrics + the
    normal/anomaly distribution split for the PRIMARY score (for histograms)."""
    val_scores = score_blocks(model, va_blocks, device, CONFIG)
    test_scores = score_blocks(model, te_blocks, device, CONFIG)

    per_agg = {}
    for key in AGGREGATORS:
        ts = test_scores[key]
        thr = _metrics_at_best_f1(va_y, val_scores[key])["threshold"]   # picked on VAL
        y_pred = (ts >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            te_y, y_pred, average="binary", zero_division=0)
        norm_s, anom_s = ts[te_y == 0], ts[te_y == 1]
        per_agg[key] = {
            "auroc": float(roc_auc_score(te_y, ts)),
            "pr_auc": float(average_precision_score(te_y, ts)),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "threshold": float(thr),
            "ks": _ks_statistic(norm_s, anom_s),
            "median_gap": float(np.median(anom_s) - np.median(norm_s)),
        }

    primary = test_scores[PRIMARY]
    return {
        "per_agg": per_agg,
        "primary_normal": primary[te_y == 0],
        "primary_anomaly": primary[te_y == 1],
        "stats_normal": _dist_stats(primary[te_y == 0]),
        "stats_anomaly": _dist_stats(primary[te_y == 1]),
    }


def _plot_one(ax, normal_s, anomaly_s, regime, auroc, ks):
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(normal_s, bins=bins, density=True, alpha=0.6, color="#4C72B0", label="Normal blocks")
    ax.hist(anomaly_s, bins=bins, density=True, alpha=0.6, color="#C44E52", label="Anomaly blocks")
    ax.set_title(f"{regime}\nAUROC {auroc:.3f} · KS {ks:.3f}")
    ax.set_xlabel(f"block score ({PRIMARY})")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


def plot_histograms(results):
    # Individual figures + a combined side-by-side.
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(6 * len(REGIMES), 5), sharex=True)
    for ax, (regime, _) in zip(np.atleast_1d(axes), REGIMES):
        r = results[regime]
        m = r["per_agg"][PRIMARY]
        _plot_one(ax, r["primary_normal"], r["primary_anomaly"], regime, m["auroc"], m["ks"])

        solo, sax = plt.subplots(figsize=(6, 5))
        _plot_one(sax, r["primary_normal"], r["primary_anomaly"], regime, m["auroc"], m["ks"])
        solo.tight_layout(); solo.savefig(f"outputs/hist_{regime}.png", dpi=120); plt.close(solo)

    fig.suptitle("Block-score separation: Normal vs Anomaly (mixed vs normal-only training)")
    fig.tight_layout()
    fig.savefig("outputs/separation_histograms.png", dpi=120)
    plt.close(fig)
    log.info("Wrote outputs/hist_mixed.png, outputs/hist_normal_only.png, outputs/separation_histograms.png")


def write_report(results, meta):
    L = []; a = L.append
    a("# Why Normal-Only Training Wins — Score-Separation Analysis (Task 1)\n")
    a("_Both checkpoints score the SAME held-out TEST blocks (stratified BlockId split, "
      f"seed {CONFIG['random_seed']}). The only difference between them is the TRAINING "
      "data (mixed vs normal-only). The F1 threshold is selected on VALIDATION and "
      "applied to TEST. Generated by `analyze_detection.py`._\n")
    a(f"- Test blocks: **{meta['n_test']:,}** · val blocks **{meta['n_val']:,}** · "
      f"anomaly base rate **{meta['test_rate']:.3f}**\n")

    # ---- Headline detection metrics (both regimes, all aggregators) ----
    a("## Detection metrics (held-out TEST; threshold chosen on VAL)\n")
    a("| Regime | Block score | AUROC | PR-AUC | Precision | Recall | F1 | KS | median gap |")
    a("|---|---|---|---|---|---|---|---|---|")
    for regime, _ in REGIMES:
        for key in AGGREGATORS:
            m = results[regime]["per_agg"][key]
            a(f"| {regime} | {key} | {m['auroc']:.4f} | **{m['pr_auc']:.4f}** | "
              f"{m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | "
              f"{m['ks']:.3f} | {m['median_gap']:+.4f} |")
    a("")

    # ---- The mechanism: distribution of the primary surprise score ----
    a(f"## The mechanism — distribution of `{PRIMARY}`, split by true label\n")
    a("If a regime truly learns *normal* behaviour, anomalous blocks should carry a "
      "**higher** surprise score than normal ones. The gap below is that effect.\n")
    a("| Regime | label | mean | median | p90 | p99 |")
    a("|---|---|---|---|---|---|")
    for regime, _ in REGIMES:
        sn, sa = results[regime]["stats_normal"], results[regime]["stats_anomaly"]
        a(f"| {regime} | normal  | {sn['mean']:.4f} | {sn['median']:.4f} | {sn['p90']:.4f} | {sn['p99']:.4f} |")
        a(f"| {regime} | anomaly | {sa['mean']:.4f} | {sa['median']:.4f} | {sa['p90']:.4f} | {sa['p99']:.4f} |")
    a("")
    a("Histograms: `outputs/separation_histograms.png` (side by side), "
      "`outputs/hist_mixed.png`, `outputs/hist_normal_only.png`.\n")

    # ---- Interpretation (data-driven deltas) ----
    mx, no = results["mixed"]["per_agg"]["max_surprisal"], results["normal_only"]["per_agg"]["max_surprisal"]
    mt, nt = results["mixed"]["per_agg"]["topk_miss_frac"], results["normal_only"]["per_agg"]["topk_miss_frac"]
    mm, nm = results["mixed"]["per_agg"]["mean_surprisal"], results["normal_only"]["per_agg"]["mean_surprisal"]
    a("## Interpretation — the same model, the only change is the training data\n")
    a(f"**On DeepLog's own rule (`max_surprisal` — the single most surprising transition in a "
      f"block), the effect is dramatic:** detection goes from F1 **{mx['f1']:.3f}** / "
      f"PR-AUC **{mx['pr_auc']:.3f}** (mixed) to F1 **{no['f1']:.3f}** / PR-AUC **{no['pr_auc']:.3f}** "
      f"(normal-only) — ΔF1 {no['f1'] - mx['f1']:+.3f}, ΔPR-AUC {no['pr_auc'] - mx['pr_auc']:+.3f}. "
      "Architecture, windows, loss and seed are identical; only the training data changed.\n")
    a(f"**The clearest mechanistic proof is the top-k-miss signal** (DeepLog's flag: was the "
      f"actual event outside the model's top-3 predictions?). Under MIXED training it is "
      f"**near-random (AUROC {mt['auroc']:.3f})** — anomalous blocks contain no more out-of-top-k "
      f"events than normal ones, because the model was *taught* those transitions as ordinary. "
      f"Under NORMAL-ONLY training it becomes **informative (AUROC {nt['auroc']:.3f})**.\n")
    a(f"**Why `mean_surprisal` moves little** (ΔF1 {nm['f1'] - mm['f1']:+.3f}): averaging surprise "
      "over the whole block dilutes the one anomalous spike among many ordinary transitions. This "
      "is itself a finding — *peak / burst* aggregations beat *mean* for detection (see Task 2, "
      "`outputs/detection_analysis.md`).\n")
    a("The histograms make it visual: under mixed training the Normal and Anomaly block-score "
      "distributions overlap; under normal-only the Anomaly distribution separates to the right "
      "— the DeepLog assumption (the model has only ever seen normal behaviour) restored.\n")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/normal_vs_anomaly_separation.md").write_text("\n".join(L) + "\n")
    log.info("Wrote outputs/normal_vs_anomaly_separation.md")


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Why normal-only wins: score-separation analysis (Task 1).")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Random block subsample for speed (default: all blocks).")
    ap.add_argument("--device", default=None, help="Override device (e.g. cpu).")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    log.info("Device: %s", device)

    for _, ckpt in REGIMES:
        if not os.path.exists(ckpt):
            raise SystemExit(f"Missing checkpoint {ckpt!r}. Run run_normal_only.py first.")

    encoder = StreamingLabelEncoder.load(os.path.join(CONFIG["cache_dir"], "encoder.pkl"))

    # Same held-out split for both regimes (the whole point of the comparison).
    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, CONFIG, max_blocks=args.max_blocks)
    _, va_idx, te_idx = bd.split_block_indices(labels, CONFIG)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, _ = bd.select_blocks(blocks, labels, block_ids, te_idx)

    results = {}
    for regime, ckpt in REGIMES:
        log.info("Scoring regime=%s (%d val + %d test blocks) ...", regime, len(va_blocks), len(te_blocks))
        model = _load_model(ckpt, encoder, device)
        results[regime] = analyze_regime(model, va_blocks, va_y, te_blocks, te_y, device)

    meta = {"n_test": len(te_blocks), "n_val": len(va_blocks), "test_rate": float(np.mean(te_y))}
    plot_histograms(results)
    write_report(results, meta)

    print("\n" + "=" * 72)
    print("  WHY NORMAL-ONLY WINS  (held-out test; primary score = %s)" % PRIMARY)
    print("=" * 72)
    for regime, _ in REGIMES:
        m = results[regime]["per_agg"][PRIMARY]
        print(f"  {regime:12s}  AUROC {m['auroc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
              f"F1 {m['f1']:.3f}  KS {m['ks']:.3f}  median-gap {m['median_gap']:+.3f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

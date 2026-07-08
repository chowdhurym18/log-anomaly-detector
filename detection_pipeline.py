# =============================================================================
# pipeline/detection_pipeline.py — the PRODUCTION suspicious-activity detector.
#
# It reframes the system
# from "next-event prediction" into:
#
#     Logs → GRU (normal-only) → Surprise score → Confidence
#          → Normal / Uncertain / Suspicious → Human explanation
#
# WHY normal-only: a GRU trained on NORMAL blocks only stays *surprised* by
# anomalous transitions (the DeepLog assumption). The mixed-data model learned
# anomalies as ordinary and barely flags them (see analyze_detection.py / Task 1).
#
# Design (reuses existing utilities — no new model, no new windowing):
#   * Detector  : the normal-only block checkpoint (config["detection_checkpoint"]).
#   * Score     : per-window surprise aggregated to the block, picked by the Task-2
#                 study (config["detection_score"]); computed via
#                 run_scoring_comparison.{per_window_signals,build_scores}.
#   * Bands     : BOTH cutoffs are DERIVED FROM VALIDATION (labels available), never
#                 guessed — suspicious = best-F1 operating point; uncertain = a
#                 high-recall point below which a block is confidently normal.
#   * Explain   : the most-surprising window of each flagged block is handed to the
#                 existing hallucination-proof llm.explanation.generate_explanation.
#
# "Suspicious" is a PRESENTATION label; the internal enum stays ANOMALY/UNCERTAIN/
# NORMAL so existing CSV consumers and tests are unaffected.
#
# Output: outputs/detection_report.md (human-readable) + outputs/detection_results.csv
# =============================================================================

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

# Allow running as a bare script (python pipeline/detection_pipeline.py) — put the
# repo root on sys.path so the top-level imports below resolve. Mirrors anomaly_eval.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
)

from config import CONFIG, variant_flags
from utils.format import format_pct as _pct, band_label
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from evaluation.anomaly_eval import _metrics_at_best_f1
from run_scoring_comparison import per_window_signals, build_scores
from llm.explanation import generate_explanation

log = logging.getLogger(__name__)

# Internal enum → emoji badge. The human-facing band name comes from
# utils.format.band_label; _pct is utils.format.format_pct (imported above). The
# internal ANOMALY/UNCERTAIN/NORMAL enum is kept for the CSV + tests (back-compat).
BADGE = {"ANOMALY": "⚠ ", "UNCERTAIN": "❓", "NORMAL": "✅"}


def _suspicion_interpretation(band: str) -> str:
    return {
        "ANOMALY":   "high surprise — flagged as SUSPICIOUS for investigation.",
        "UNCERTAIN": "moderate surprise — routed for additional review.",
        "NORMAL":    "low surprise — consistent with normal behaviour.",
    }[band]


# ---------------------------------------------------------------------------
# Band cutoffs — BOTH derived from validation data (Task 3 requirement).
# ---------------------------------------------------------------------------
def derive_bands(val_score: np.ndarray, val_y: np.ndarray, max_normal_rate: float = 0.01):
    """Return (uncertain_thr, suspicious_thr), both learned on VALIDATION.

      * suspicious_thr = the best-F1 operating point on val (precision-leaning) —
                         at/above it a block is flagged SUSPICIOUS.
      * uncertain_thr  = the top of the largest "confidently normal" band: the
                         highest cut below which validation blocks were at most
                         `max_normal_rate` anomalous (a controlled false-clear rate).
                         This keeps the Normal band as large as possible while
                         staying low-risk — unlike a fixed recall target, which on
                         an overlapping tail collapses the Normal band to nothing.
    Clamped so uncertain_thr <= suspicious_thr.
    """
    suspicious_thr = _metrics_at_best_f1(val_y, val_score)["threshold"]
    order = np.argsort(val_score, kind="stable")          # ascending by surprise
    ys = val_y[order].astype(np.float64)
    ss = val_score[order]
    cum_rate = np.cumsum(ys) / (np.arange(len(ys)) + 1.0)  # anomaly rate of each low-score prefix
    ok = np.where(cum_rate <= max_normal_rate)[0]
    if len(ok):
        k = int(ok.max())                                  # Normal band = the lowest k+1 scores
        uncertain_thr = float(ss[min(k + 1, len(ss) - 1)])
    else:
        uncertain_thr = float(ss[0])                       # no clean Normal band possible
    uncertain_thr = min(uncertain_thr, suspicious_thr)
    return uncertain_thr, suspicious_thr


def classify(score: np.ndarray, uncertain_thr: float, suspicious_thr: float) -> np.ndarray:
    out = np.full(len(score), "NORMAL", dtype=object)
    out[score >= uncertain_thr] = "UNCERTAIN"
    out[score >= suspicious_thr] = "ANOMALY"
    return out


# ---------------------------------------------------------------------------
# Explanation support — find the single most-surprising window of a block.
# ---------------------------------------------------------------------------
def most_surprising_window(model, block, device, config):
    """Return the block's peak-surprise window: its context, the actual next event,
    the model's predicted next event, the model's confidence (MSP) and the
    surprisal (1 - P(actual)). Used to ground a per-block explanation."""
    seq_len = config["sequence_length"]
    best = None
    model.eval()
    with torch.no_grad():
        for i in range(1, len(block)):
            ctx = np.asarray(block[max(0, i - seq_len):i], dtype=np.int64)
            tgt = int(block[i])
            probs = torch.softmax(model(torch.from_numpy(ctx[None, :]).to(device)), dim=1)
            probs = probs.cpu().numpy()[0]
            surprisal = 1.0 - float(probs[tgt])
            if best is None or surprisal > best["surprisal"]:
                best = {"context": ctx.tolist(), "target": tgt,
                        "pred": int(probs.argmax()), "msp": float(probs.max()),
                        "surprisal": surprisal}
    return best


def _explain_block(model, block, band, device, config, encoder, event_context):
    w = most_surprising_window(model, block, device, config)
    seq = encoder.inverse_transform(w["context"], context="detection explanation context")
    pred_ev = encoder.inverse_transform([w["pred"]], context="detection pred")[0]
    actual_ev = encoder.inverse_transform([w["target"]], context="detection actual")[0]
    text = generate_explanation(
        sequence=seq,
        predicted_event=pred_ev,
        actual_event=actual_ev,
        # Convention (matches anomaly_pipeline): confidence = 1 - anomaly_score. Here
        # anomaly_score is the surprise on the ACTUAL event, so confidence is the
        # probability the model gave that event — correctly LOW when it was surprising.
        confidence=1.0 - w["surprisal"],
        anomaly_score=w["surprisal"],   # surprise on the ACTUAL event [0,1]
        classification=band,            # internal enum (ANOMALY/UNCERTAIN)
        event_context=event_context,
        use_llm=config.get("detection_use_llm", True),
    )
    return w, text


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _write_report(meta, bands, test_y, test_score, unc, susp, det, examples, score_name,
                  out_dir="outputs", block_desc="one HDFS operation's full event sequence"):
    n = len(bands)
    L = []; a = L.append
    a("# Suspicious-Activity Detection Report\n")
    a("_The system learns NORMAL behaviour, measures how SURPRISING each block of "
      "activity is, turns that into a confidence score, and sorts every block into "
      "**Normal**, **Uncertain**, or **Suspicious**. Suspicious and uncertain blocks "
      "get a plain-language explanation. Generated by `pipeline/detection_pipeline.py`._\n")

    a("## What the system decided\n")
    a(f"It reviewed **{n:,}** blocks of activity (a *block* is {block_desc}). Of those:\n")
    a("| Band | What it means | Blocks | Share | Actually anomalous |")
    a("|---|---|---|---|---|")
    for b in ("NORMAL", "UNCERTAIN", "ANOMALY"):
        mask = bands == b
        cnt = int(mask.sum())
        share = cnt / max(n, 1)
        purity = float(test_y[mask].mean()) if cnt else 0.0
        a(f"| {BADGE[b]} {band_label(b)} | {_suspicion_interpretation(b)} | {cnt:,} | "
          f"{_pct(share)} | {_pct(purity)} |")
    a("")
    a("_'Actually anomalous' uses the ground-truth labels only to grade the system — the "
      "detector itself never sees them. A good detector keeps the Normal band near 0% and "
      "concentrates the true anomalies in the Suspicious band._\n")

    a("## How well it detects (held-out test, graded against ground truth)\n")
    a(f"- Suspicious-activity **F1: {_pct(det['f1'])}**  (Precision {_pct(det['precision'])} · "
      f"Recall {_pct(det['recall'])})")
    a(f"- Ranking quality — **PR-AUC {det['pr_auc']:.3f}** (anomalies are only "
      f"{_pct(meta['test_rate'])} of blocks, so this is the metric that matters) · "
      f"AUROC {det['auroc']:.3f}")
    a(f"- Surprise signal used: `{score_name}`.\n")

    a("## How the thresholds were set (no guessing)\n")
    a("Both cut-offs are **learned from a separate validation set**, never hand-picked:\n")
    a(f"- **Suspicious** when the block's surprise is at/above the best-F1 operating point "
      f"(threshold = {susp:.4f}).")
    a(f"- **Normal** when surprise is below the auto-clear boundary (threshold = {unc:.4f}) — "
      "the largest band of lowest-surprise blocks that validation showed to be almost never "
      "anomalous (see the 'Actually anomalous' column above), so clearing them is low-risk.")
    a("- **Uncertain** is everything in between — the ambiguous middle that earns a closer look.\n")

    if examples:
        a("## Example explanations (most suspicious blocks)\n")
        a("_Every verdict field below is computed deterministically; the LLM only writes the "
          "single 'Evidence-Based Interpretation' line, which is scrubbed against the facts._\n")
        for k, (w, text, block_score) in enumerate(examples, 1):
            a(f"### Suspicious block #{k}\n")
            a(f"**Why this block was flagged:** its surprise score (`{score_name}`) was "
              f"**{block_score:.3f}**, at or above the **{susp:.3f}** Suspicious threshold learned "
              f"on validation. The single most-surprising transition inside the block — "
              f"surprise **{_pct(w['surprisal'])}** on the event that actually occurred — is "
              "explained below.\n")
            a("```")
            a(text)
            a("```")
            a("")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(out_dir) / "detection_report.md"
    report_path.write_text("\n".join(L) + "\n")
    log.info("Wrote %s", report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# How the report describes one "block" — sessionisation differs per dataset.
_BLOCK_DESC = {
    "hdfs": "one HDFS operation's full event sequence",
    "bgl":  "a fixed 100-line window of consecutive BGL log lines",
}


def run_detection_pipeline(encoder, device, config,
                           event_context: Optional[Dict[str, str]] = None,
                           max_blocks: Optional[int] = None,
                           out_dir: str = "outputs"):
    """Score held-out blocks with the normal-only detector, sort them into
    Normal/Uncertain/Suspicious using validation-derived thresholds, explain the
    most suspicious ones, and write a human-readable report + CSV."""
    ckpt = config.get("detection_checkpoint", ".checkpoints/normal_only/normal_only/best_model.pt")
    score_name = config.get("detection_score", "nll_-logp_mean")
    if not Path(ckpt).exists():
        raise SystemExit(
            f"Detection checkpoint {ckpt!r} not found. Train the normal-only detector "
            "first:  venv/bin/python run_normal_only.py")

    flags = variant_flags("gru")
    model = GRUAnomalyDetector(
        len(encoder.classes_), config["embedding_dim"], config["hidden_dim"],
        config["num_layers"], config["dropout"], **flags).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log.info("Detector: %s  (normal-only GRU)", ckpt)

    # Held-out split: derive thresholds on VAL, decide on TEST (never in-sample).
    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, config, max_blocks=max_blocks)
    _, va_idx, te_idx = bd.split_block_indices(labels, config)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, te_ids = bd.select_blocks(blocks, labels, block_ids, te_idx)

    log.info("Scoring %d val + %d test blocks with `%s` ...", len(va_blocks), len(te_blocks), score_name)
    val_sig = per_window_signals(model, va_blocks, device, config)
    test_sig = per_window_signals(model, te_blocks, device, config)
    roll = config.get("detection_roll_window", 5)
    val_scores = build_scores(val_sig, ref_sig=val_sig, roll_window=roll)
    test_scores = build_scores(test_sig, ref_sig=val_sig, roll_window=roll)
    if score_name not in test_scores:
        raise SystemExit(f"detection_score {score_name!r} not in {list(test_scores)}")
    val_score, test_score = val_scores[score_name], test_scores[score_name]

    # Bands from validation; classify test.
    unc, susp = derive_bands(val_score, va_y, config.get("detection_max_normal_rate", 0.01))
    bands = classify(test_score, unc, susp)

    # Detection metrics at the suspicious threshold.
    y_pred = (test_score >= susp).astype(int)
    p, r, f, _ = precision_recall_fscore_support(te_y, y_pred, average="binary", zero_division=0)
    det = {"precision": float(p), "recall": float(r), "f1": float(f),
           "auroc": float(roc_auc_score(te_y, test_score)),
           "pr_auc": float(average_precision_score(te_y, test_score))}

    log.info("Bands — Normal %d | Uncertain %d | Suspicious %d  (F1 %.3f, PR-AUC %.3f)",
             int((bands == "NORMAL").sum()), int((bands == "UNCERTAIN").sum()),
             int((bands == "ANOMALY").sum()), det["f1"], det["pr_auc"])

    # Explain the top-N most suspicious blocks.
    max_expl = config.get("detection_max_explanations", 10)
    susp_idx = np.where(bands == "ANOMALY")[0]
    susp_idx = susp_idx[np.argsort(test_score[susp_idx])[::-1]][:max_expl]
    examples = []
    for i in susp_idx:
        w, text = _explain_block(model, te_blocks[i], "ANOMALY", device, config, encoder, event_context)
        examples.append((w, text, float(test_score[i])))   # carry the block score for "why flagged"

    meta = {"n_test": len(te_blocks), "n_val": len(va_blocks), "test_rate": float(np.mean(te_y))}
    block_desc = _BLOCK_DESC.get(config.get("dataset", "hdfs"), "one session's event sequence")
    _write_report(meta, bands, te_y, test_score, unc, susp, det, examples, score_name,
                  out_dir=out_dir, block_desc=block_desc)

    # CSV (internal enum retained for backward compatibility).
    csv_path = os.path.join(out_dir, "detection_results.csv")
    pd.DataFrame({
        "block_id": te_ids,
        "n_events": [len(b) for b in te_blocks],
        "surprise_score": np.round(test_score, 6),
        "classification": bands,          # ANOMALY / UNCERTAIN / NORMAL
        "true_label": te_y,
    }).to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)
    return det


def main():
    import argparse
    from utils.logging_utils import setup_logging
    from config import get_device, config_for_dataset
    from preprocessing.templates import load_event_templates

    setup_logging()
    ap = argparse.ArgumentParser(description="Suspicious-activity detection pipeline.")
    ap.add_argument("--max-blocks", type=int, default=None, help="Random subsample for speed.")
    ap.add_argument("--no-llm", action="store_true", help="Skip Ollama; deterministic report only.")
    ap.add_argument("--dataset", default=None,
                    help="Dataset name from CONFIG['datasets'] (default: CONFIG['dataset'] = 'hdfs').")
    args = ap.parse_args()

    cfg = config_for_dataset(args.dataset)
    if args.no_llm:
        cfg["detection_use_llm"] = False
    device = get_device()
    log.info("Device: %s", device)
    encoder = StreamingLabelEncoder.load(f"{cfg['cache_dir']}/encoder.pkl")
    event_map = load_event_templates(cfg["datasets"][cfg["dataset"]]["templates"])
    run_detection_pipeline(encoder, device, cfg, event_context=event_map,
                           max_blocks=args.max_blocks,
                           out_dir=cfg.get("output_dir", "outputs"))


if __name__ == "__main__":
    main()

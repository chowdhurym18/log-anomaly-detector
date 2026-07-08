# =============================================================================
# run_bgl_evaluation.py — Tasks 2 & 5: the full MEASURED BGL evaluation + the
# BGL↔HDFS comparison. Writes outputs/bgl_results.md and outputs/bgl_vs_hdfs.md.
#
# Honesty rules baked in:
#   * Both datasets are evaluated through the IDENTICAL block-level protocol
#     (same split function, same production detector checkpoint, same
#     val-derived thresholds), so every side-by-side number is like-for-like.
#   * Every number is measured in this run, except training wall-clock, which
#     is parsed from the measured-cost table that run_normal_only.py wrote
#     during the timed training runs (provenance is stated in the reports).
#   * Chronological-split robustness check (BGL only): BGL blocks are fixed
#     100-line windows in file/time order, so a first-70/10/20 split answers
#     "does the random split inflate results via temporally-correlated
#     near-duplicate windows?" (the Le & Zhang ICSE'22 critique). A drop here
#     is reported, not hidden.
#
# Prereqs: timed training runs done —
#   venv/bin/python run_normal_only.py --dataset bgl --epochs 8 --patience 3
#   venv/bin/python run_normal_only.py --dataset hdfs --regimes normal_only \
#       --epochs 8 --patience 3 --checkpoint-dir .checkpoints/timing_hdfs
# Run:  venv/bin/python run_bgl_evaluation.py [--skip-chrono] [--skip-llm]
# =============================================================================

import os
import re
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    confusion_matrix,
)

from config import CONFIG, get_device, variant_flags, config_for_dataset
from utils.logging_utils import setup_logging
from utils.format import format_pct as pct
from utils.perf import Timer, peak_rss_mb, device_sync
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing.templates import load_event_templates
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from run_normal_only import train_on_buckets, _fresh_model
from run_scoring_comparison import per_window_signals, build_scores
from pipeline.detection_pipeline import derive_bands, classify, most_surprising_window
from llm.ollama_client import test_ollama_connection
from llm.uncertain_classifier import classify_uncertain

log = logging.getLogger(__name__)

BAND_NAMES = {"NORMAL": "Normal", "UNCERTAIN": "Uncertain", "ANOMALY": "Suspicious"}


# ---------------------------------------------------------------------------
# Next-event metrics on the block test split (identical protocol per dataset).
# ---------------------------------------------------------------------------
def next_event_metrics(model, blocks, device, config, ks=(1, 3, 5)):
    seq_len = config["sequence_length"]
    buckets = bd.build_window_buckets(blocks, seq_len)
    kmax = max(ks)
    preds, trues = [], []
    hits = {k: [] for k in ks}
    model.eval()
    with torch.no_grad():
        for _ln, (ctx, tgt) in buckets.items():
            for s in range(0, len(tgt), 4096):
                bx = torch.from_numpy(ctx[s:s + 4096]).long().to(device)
                top = model(bx).topk(kmax, dim=1).indices.cpu().numpy()
                bt = tgt[s:s + 4096]
                preds.append(top[:, 0]); trues.append(bt)
                for k in ks:
                    hits[k].append((top[:, :k] == bt[:, None]).any(axis=1))
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    out = {"n_windows": int(len(trues)),
           "n_classes_seen": int(len(np.unique(trues)))}
    for k in ks:
        out[f"top{k}"] = float(np.concatenate(hits[k]).mean())
    for avg in ("weighted", "macro"):
        p, r, f, _ = precision_recall_fscore_support(
            trues, preds, average=avg, zero_division=0)
        out[f"{avg}_precision"] = float(p)
        out[f"{avg}_recall"] = float(r)
        out[f"{avg}_f1"] = float(f)
    return out


# ---------------------------------------------------------------------------
# Production detection + routing, instrumented (throughput, separation, bands).
# ---------------------------------------------------------------------------
def detection_eval(model, va_blocks, va_y, te_blocks, te_y, device, cfg):
    roll = cfg.get("detection_roll_window", 5)
    score_name = cfg.get("detection_score", "nll_-logp_mean")

    with Timer() as t_score:
        val_sig = per_window_signals(model, va_blocks, device, cfg)
        test_sig = per_window_signals(model, te_blocks, device, cfg)
        device_sync(device)
    val_score = build_scores(val_sig, ref_sig=val_sig, roll_window=roll)[score_name]
    test_score = build_scores(test_sig, ref_sig=val_sig, roll_window=roll)[score_name]

    unc, susp = derive_bands(val_score, va_y, cfg.get("detection_max_normal_rate", 0.01))
    bands = classify(test_score, unc, susp)
    y_pred = (test_score >= susp).astype(int)
    p, r, f, _ = precision_recall_fscore_support(te_y, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(te_y, y_pred, labels=[0, 1]).ravel()

    s_norm, s_anom = test_score[te_y == 0], test_score[te_y == 1]
    n_scored = len(va_blocks) + len(te_blocks)
    n_windows = len(val_sig["nll"]) + len(test_sig["nll"])

    routing = {}
    for bkey in ("NORMAL", "UNCERTAIN", "ANOMALY"):
        mask = bands == bkey
        cnt = int(mask.sum())
        routing[bkey] = {"count": cnt, "share": cnt / max(len(bands), 1),
                         "purity": float(te_y[mask].mean()) if cnt else 0.0}

    def q(x, p_):
        return float(np.percentile(x, p_)) if len(x) else float("nan")

    return {
        "score_name": score_name,
        "precision": float(p), "recall": float(r), "f1": float(f),
        "auroc": float(roc_auc_score(te_y, test_score)),
        "pr_auc": float(average_precision_score(te_y, test_score)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "thr_uncertain": float(unc), "thr_suspicious": float(susp),
        "routing": routing,
        "llm_share": routing["UNCERTAIN"]["share"] + routing["ANOMALY"]["share"],
        "ks": float(ks_2samp(s_norm, s_anom).statistic) if len(s_anom) and len(s_norm) else float("nan"),
        "median_gap": (float(np.median(s_anom) - np.median(s_norm))
                       if len(s_anom) and len(s_norm) else float("nan")),
        "dist": {"normal": {"mean": float(s_norm.mean()), "median": q(s_norm, 50),
                            "p90": q(s_norm, 90), "p99": q(s_norm, 99)},
                 "anomaly": {"mean": float(s_anom.mean()) if len(s_anom) else float("nan"),
                             "median": q(s_anom, 50), "p90": q(s_anom, 90), "p99": q(s_anom, 99)}},
        "score_s": t_score.seconds,
        "blocks_per_s": n_scored / t_score.seconds if t_score.seconds > 0 else 0.0,
        "windows_per_s": n_windows / t_score.seconds if t_score.seconds > 0 else 0.0,
        "n_scored_blocks": n_scored, "n_windows": n_windows,
        "bands": bands, "test_score": test_score,
    }


# ---------------------------------------------------------------------------
# Measured Llama latency on the blocks production actually routes to it.
# ---------------------------------------------------------------------------
def measure_llm_latency(model, te_blocks, det, device, cfg, encoder, event_map, n=5):
    ok, _, _ = test_ollama_connection()
    if not ok:
        log.warning("Ollama unreachable — LLM latency not measured.")
        return None
    idx = np.where(det["bands"] != "NORMAL")[0]
    if len(idx) == 0:
        return None
    idx = idx[np.argsort(det["test_score"][idx])[::-1][:n]]
    lats = []
    for i in idx:
        w = most_surprising_window(model, te_blocks[i], device, cfg)
        seq = encoder.inverse_transform(w["context"], context="latency ctx")
        pred = encoder.inverse_transform([w["pred"]], context="latency pred")[0]
        actual = encoder.inverse_transform([w["target"]], context="latency actual")[0]
        r = classify_uncertain(seq, pred, actual, event_map,
                               1.0 - w["surprisal"], w["surprisal"], temperature=0.0)
        lats.append(r["latency"])
    return {"n": len(lats), "avg_s": float(np.mean(lats)), "median_s": float(np.median(lats))}


# ---------------------------------------------------------------------------
# Chronological-split robustness check (BGL): first 70/10/20 in time order.
# ---------------------------------------------------------------------------
def chrono_split_eval(blocks, labels, encoder, device, cfg, epochs, patience):
    n = len(blocks)
    i_tr, i_va = int(n * 0.7), int(n * 0.8)
    tr_blocks, tr_y = blocks[:i_tr], labels[:i_tr]
    va_blocks, va_y = blocks[i_tr:i_va], labels[i_tr:i_va]
    te_blocks, te_y = blocks[i_va:], labels[i_va:]
    norm_blocks = [b for b, y in zip(tr_blocks, tr_y) if y == 0]
    log.info("Chrono split — train %d (%d normal) | val %d (%.1f%% anom) | test %d (%.1f%% anom)",
             len(tr_blocks), len(norm_blocks), len(va_blocks), 100 * float(np.mean(va_y)),
             len(te_blocks), 100 * float(np.mean(te_y)))

    seq_len = cfg["sequence_length"]
    vocab = len(encoder.classes_)
    buckets = bd.build_window_buckets(norm_blocks, seq_len)
    val_buckets = bd.build_window_buckets(va_blocks, seq_len)
    all_tgt = np.concatenate([t for _, t in buckets.values()])
    cw = bd.sqrt_inverse_class_weights(all_tgt, vocab) if cfg["use_class_weights"] else None
    model = _fresh_model(encoder, device, cfg)
    ckpt = os.path.join(cfg["block_checkpoint_dir"], "chrono", "best_model.pt")
    with Timer() as t_train:
        model = train_on_buckets(model, buckets, val_buckets, device, cfg,
                                 epochs=epochs, ckpt_path=ckpt, class_weights=cw,
                                 patience=patience)
        device_sync(device)

    det = detection_eval(model, va_blocks, np.asarray(va_y), te_blocks, np.asarray(te_y),
                         device, cfg)
    det["train_s"] = t_train.seconds
    det["split"] = {"n_train": len(tr_blocks), "n_train_normal": len(norm_blocks),
                    "n_val": len(va_blocks), "n_test": len(te_blocks),
                    "val_rate": float(np.mean(va_y)), "test_rate": float(np.mean(te_y))}
    return det


# ---------------------------------------------------------------------------
# Training-cost provenance: parse run_normal_only.py's measured-cost table.
# ---------------------------------------------------------------------------
_COST_ROW = re.compile(
    r"\|\s*(mixed|normal_only)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


def parse_training_costs(report_path):
    path = Path(report_path)
    if not path.exists():
        return {}
    out = {}
    for m in _COST_ROW.finditer(path.read_text()):
        out[m.group(1)] = {
            "train_s": float(m.group(2)), "epochs": int(m.group(3)),
            "score_s": float(m.group(4)), "blocks_per_s": float(m.group(5)),
            "peak_rss_mb": float(m.group(6)),
        }
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _routing_table(det, a):
    a("| Band | Blocks | Share | Actually anomalous |")
    a("|---|---|---|---|")
    for bkey in ("NORMAL", "UNCERTAIN", "ANOMALY"):
        r = det["routing"][bkey]
        a(f"| {BAND_NAMES[bkey]} | {r['count']:,} | {pct(r['share'])} | {pct(r['purity'])} |")
    a("")


def _confusion_table(det, a):
    c = det["confusion"]
    a("|  | Predicted Normal | Predicted Suspicious |")
    a("|---|---|---|")
    a(f"| **Actually Normal** | TN = {c['tn']:,} | FP = {c['fp']:,} |")
    a(f"| **Actually Anomalous** | FN = {c['fn']:,} | TP = {c['tp']:,} |")
    a("")


def write_bgl_results(r, chrono, costs, out_path="outputs/bgl_results.md"):
    b, nxt, det, meta = r["bgl"], r["bgl"]["next_event"], r["bgl"]["detection"], r["bgl"]["meta"]
    L = []; a = L.append
    a("# BGL Results — Suspicious-Activity Detection (Task 2)\n")
    a("_A FRESH normal-only GRU trained on BGL (BlueGene/L supercomputer logs) and run "
      "through the production pipeline: surprise scoring → validation-derived "
      "Normal/Uncertain/Suspicious bands → Llama explanation for flagged blocks. All "
      "numbers below are measured; the training cost comes from the timed "
      "`run_normal_only.py --dataset bgl` run. Generated by `run_bgl_evaluation.py`._\n")

    a("## Dataset & setup\n")
    a(f"- **{meta['vocab']:,}** Drain3 event templates (vs HDFS's 29); blocks are fixed "
      "**100-line** windows in time order; a block is anomalous if it contains ≥1 alert line.")
    a(f"- Blocks: **{meta['n_blocks']:,}** total — train {meta['n_train']:,} "
      f"({meta['n_train_normal']:,} normal used for training) · val {meta['n_val']:,} · "
      f"test {meta['n_test']:,}; test anomaly base rate **{pct(meta['test_rate'])}** "
      "(stratified random split, seed 42 — same protocol as HDFS).")
    a(f"- Detector: `{r['bgl']['checkpoint']}` · surprise score `{det['score_name']}` · "
      "thresholds learned on validation only.\n")

    a("## Detection performance (held-out test)\n")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| **Detection F1** | **{det['f1']:.4f}** |")
    a(f"| Precision | {det['precision']:.4f} |")
    a(f"| Recall | {det['recall']:.4f} |")
    a(f"| PR-AUC | {det['pr_auc']:.4f} |")
    a(f"| AUROC | {det['auroc']:.4f} |")
    a(f"| Suspicious threshold (val best-F1) | {det['thr_suspicious']:.4f} |")
    a(f"| Auto-clear threshold (val ≤1% anomalous prefix) | {det['thr_uncertain']:.4f} |")
    a("")
    a("### Confusion matrix (at the val-derived Suspicious threshold)\n")
    _confusion_table(det, a)

    a("## Routing distribution (test blocks)\n")
    _routing_table(det, a)
    a(f"**LLM share (Uncertain + Suspicious): {pct(det['llm_share'])}** — only these blocks "
      "ever reach Llama; the Normal band is auto-cleared.\n")
    if r["bgl"].get("llm"):
        llm = r["bgl"]["llm"]
        n_test = meta["n_test"]
        llm_blocks = int(round(det["llm_share"] * n_test))
        a(f"- Measured Llama latency: **{llm['avg_s']:.2f} s/call** (median {llm['median_s']:.2f}s, "
          f"n={llm['n']} flagged blocks, llama3.2:1b, temperature 0).")
        a(f"- Cost projection with the measured latency: explaining ALL {n_test:,} test blocks "
          f"≈ {n_test * llm['avg_s'] / 3600:.1f} h; routing only the flagged {llm_blocks:,} "
          f"≈ {llm_blocks * llm['avg_s'] / 3600:.1f} h — a "
          f"**{(1.0 / max(det['llm_share'], 1e-9)):.0f}× reduction**.\n")

    a("## Next-event prediction (same test blocks, same windows as the detector)\n")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Top-1 accuracy | {pct(nxt['top1'])} |")
    a(f"| Top-3 accuracy | {pct(nxt['top3'])} |")
    a(f"| Top-5 accuracy | {pct(nxt['top5'])} |")
    a(f"| Weighted F1 | {nxt['weighted_f1']:.4f} |")
    a(f"| Weighted precision / recall | {nxt['weighted_precision']:.4f} / {nxt['weighted_recall']:.4f} |")
    a(f"| Macro F1 | {nxt['macro_f1']:.4f} |")
    a(f"| Macro precision / recall | {nxt['macro_precision']:.4f} / {nxt['macro_recall']:.4f} |")
    a(f"| Test windows | {nxt['n_windows']:,} ({nxt['n_classes_seen']:,} distinct target events) |")
    a("")
    a("_Macro-F1 averages over every one of the ~1.8k templates equally; most are rare "
      "(and anomalous templates are unseen in normal-only training by design), so a low "
      "macro-F1 is expected and is NOT the detection objective — surprise on rare/unseen "
      "templates is exactly the anomaly signal._\n")
    a("_Protocol note: these next-event numbers use BLOCK-RESPECTING windows (same "
      "windowing as detection training/scoring — matches `run_generalization.py`'s BGL "
      "numbers), not the FLAT/global sliding-window protocol `hdfs_baseline_report.md` uses "
      "for HDFS (2.24M windows crossing block boundaries). The two protocols measure "
      "genuinely different things and are not meant to be numerically compared._\n")

    a("## Score separation (normal vs anomalous test blocks)\n")
    a(f"- Kolmogorov–Smirnov statistic: **{det['ks']:.3f}** · median gap "
      f"(anomaly − normal): **{det['median_gap']:.3f}**\n")
    a("| Label | Mean | Median | p90 | p99 |")
    a("|---|---|---|---|---|")
    for lab in ("normal", "anomaly"):
        d = det["dist"][lab]
        a(f"| {lab} | {d['mean']:.3f} | {d['median']:.3f} | {d['p90']:.3f} | {d['p99']:.3f} |")
    a("")

    a("## Measured cost\n")
    a("| Item | Value | Provenance |")
    a("|---|---|---|")
    tc = costs.get("bgl", {}).get("normal_only")
    if tc:
        a(f"| Training (normal-only detector) | {tc['train_s']:.0f} s ({tc['epochs']} epochs) | "
          "timed `run_normal_only.py --dataset bgl` run |")
        a(f"| Peak RSS during training run | {tc['peak_rss_mb']:.0f} MiB | same run |")
    a(f"| Detection scoring | {det['score_s']:.1f} s for {det['n_scored_blocks']:,} blocks "
      f"({det['n_windows']:,} windows) = **{det['blocks_per_s']:.0f} blocks/s** | measured "
      "this run, device-synchronized |")
    if r["bgl"].get("llm"):
        a(f"| Llama latency | {r['bgl']['llm']['avg_s']:.2f} s/call | measured this run |")
    a(f"| Peak RSS (this evaluation run) | {r['peak_rss_mb']:.0f} MiB | measured this run |")
    a(f"| Device | {r['device']} | — |")
    a("")

    if chrono:
        cs = chrono["split"]
        a("## Chronological-split robustness check\n")
        a("_BGL blocks are 100-line windows in time order, so temporally-adjacent "
          "(near-duplicate) windows can straddle a RANDOM split and flatter the results "
          "(Le & Zhang, ICSE'22). Here the model is retrained on the FIRST 70% of blocks, "
          "thresholds derived on the next 10%, and evaluated on the LAST 20% — the "
          "deployment-realistic 'train on the past, detect in the future' setting._\n")
        a("| Metric | Random split | Chronological split | Δ |")
        a("|---|---|---|---|")
        for key, name in (("f1", "Detection F1"), ("precision", "Precision"),
                          ("recall", "Recall"), ("pr_auc", "PR-AUC"), ("auroc", "AUROC")):
            a(f"| {name} | {det[key]:.4f} | {chrono[key]:.4f} | {chrono[key] - det[key]:+.4f} |")
        a("")
        a(f"_Chrono test anomaly rate {pct(cs['test_rate'])} (vs {pct(meta['test_rate'])} random) — "
          "base-rate drift between periods is itself part of the deployment reality. "
          f"Chrono training: {chrono['train_s']:.0f} s._\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L) + "\n")
    log.info("Wrote %s", out_path)


def write_bgl_vs_hdfs(r, chrono, costs, out_path="outputs/bgl_vs_hdfs.md"):
    L = []; a = L.append
    a("# BGL vs HDFS — Same Pipeline, Same Protocol, Side by Side\n")
    a("_Both columns are measured by `run_bgl_evaluation.py` in the same run, through the "
      "identical block-level protocol: production normal-only detector checkpoint, "
      "stratified random split (seed 42), `nll_-logp_mean` surprise, validation-derived "
      "bands. Nothing is quoted from older reports unless labelled._\n")

    bd_, hd = r["bgl"], r["hdfs"]
    bn, hn = bd_["next_event"], hd["next_event"]
    bdet, hdet = bd_["detection"], hd["detection"]
    bm, hm = bd_["meta"], hd["meta"]

    a("## Dataset shape\n")
    a("| | BGL | HDFS |")
    a("|---|---|---|")
    a(f"| Event templates (vocab) | {bm['vocab']:,} | {hm['vocab']:,} |")
    a(f"| Blocks (total) | {bm['n_blocks']:,} | {hm['n_blocks']:,} |")
    a(f"| Block definition | fixed 100-line window | one HDFS operation (block id) |")
    a(f"| Test blocks | {bm['n_test']:,} | {hm['n_test']:,} |")
    a(f"| Test anomaly base rate | {pct(bm['test_rate'])} | {pct(hm['test_rate'])} |")
    a("")

    a("## Results (held-out test, val-derived thresholds)\n")
    a("| Metric | BGL | HDFS |")
    a("|---|---|---|")
    a(f"| Next-event Top-1 | {pct(bn['top1'])} | {pct(hn['top1'])} |")
    a(f"| Next-event Top-3 | {pct(bn['top3'])} | {pct(hn['top3'])} |")
    a(f"| Next-event Top-5 | {pct(bn['top5'])} | {pct(hn['top5'])} |")
    a(f"| Next-event Weighted F1 | {bn['weighted_f1']:.4f} | {hn['weighted_f1']:.4f} |")
    a(f"| Next-event Macro F1 | {bn['macro_f1']:.4f} | {hn['macro_f1']:.4f} |")
    a(f"| **Detection F1** | **{bdet['f1']:.4f}** | **{hdet['f1']:.4f}** |")
    a(f"| Detection Precision | {bdet['precision']:.4f} | {hdet['precision']:.4f} |")
    a(f"| Detection Recall | {bdet['recall']:.4f} | {hdet['recall']:.4f} |")
    a(f"| PR-AUC | {bdet['pr_auc']:.4f} | {hdet['pr_auc']:.4f} |")
    a(f"| AUROC | {bdet['auroc']:.4f} | {hdet['auroc']:.4f} |")
    a(f"| KS separation | {bdet['ks']:.3f} | {hdet['ks']:.3f} |")
    a("")
    a("_Next-event rows use BLOCK-RESPECTING windows for both datasets (identical protocol, "
      "so BGL vs HDFS here is apples-to-apples). This differs from `hdfs_baseline_report.md`'s "
      "HDFS next-event numbers (Top-1 82.6% for the same normal-only checkpoint), which use "
      "the FLAT/global sliding-window protocol instead — the two are different measurement "
      "protocols on the same model, not a discrepancy.\n")

    a("### Confusion matrices (val-derived Suspicious threshold)\n")
    a("**BGL**\n")
    _confusion_table(bdet, a)
    a("**HDFS**\n")
    _confusion_table(hdet, a)

    a("## Routing distribution\n")
    a("| Band | BGL share (purity) | HDFS share (purity) |")
    a("|---|---|---|")
    for bkey in ("NORMAL", "UNCERTAIN", "ANOMALY"):
        rb, rh = bdet["routing"][bkey], hdet["routing"][bkey]
        a(f"| {BAND_NAMES[bkey]} | {pct(rb['share'])} ({pct(rb['purity'])}) | "
          f"{pct(rh['share'])} ({pct(rh['purity'])}) |")
    a(f"| **LLM share** | **{pct(bdet['llm_share'])}** | **{pct(hdet['llm_share'])}** |")
    a("")

    a("## Measured cost\n")
    a("| Item | BGL | HDFS | Provenance |")
    a("|---|---|---|---|")
    tb = costs.get("bgl", {}).get("normal_only")
    th = costs.get("hdfs", {}).get("normal_only")
    if tb and th:
        a(f"| Training (normal-only, epochs≤8, patience 3) | {tb['train_s']:.0f} s "
          f"({tb['epochs']} epochs) | {th['train_s']:.0f} s ({th['epochs']} epochs) | "
          "timed matched-settings `run_normal_only.py` runs (HDFS run wrote to a throwaway "
          "checkpoint dir; production HDFS checkpoints untouched) |")
        a(f"| Peak RSS during training | {tb['peak_rss_mb']:.0f} MiB | {th['peak_rss_mb']:.0f} MiB "
          "| same runs |")
    elif tb:
        a(f"| Training (normal-only) | {tb['train_s']:.0f} s ({tb['epochs']} epochs) | "
          "not re-measured | timed BGL run; HDFS timing run missing |")
    a(f"| Detection scoring throughput | {bdet['blocks_per_s']:.0f} blocks/s "
      f"({bdet['windows_per_s']:.0f} windows/s) | {hdet['blocks_per_s']:.0f} blocks/s "
      f"({hdet['windows_per_s']:.0f} windows/s) | measured this run, device-synchronized |")
    lb, lh = bd_.get("llm"), hd.get("llm")
    lb_s = f"{lb['avg_s']:.2f}" if lb else "n/a"
    lh_s = f"{lh['avg_s']:.2f}" if lh else "n/a"
    a(f"| Llama latency (s/call) | {lb_s} | {lh_s} | measured this run |")
    a(f"| Peak RSS (evaluation run) | {r['peak_rss_mb']:.0f} MiB | (same process) | "
      "measured this run; one process evaluated both datasets |")
    a(f"| Device | {r['device']} | {r['device']} | — |")
    a("")

    a("## Honest caveats (why BGL's higher numbers do NOT mean 'BGL is solved')\n")
    a(f"- **Base rate**: BGL anomalies are {pct(bm['test_rate'])} of test blocks vs HDFS's "
      f"{pct(hm['test_rate'])}; F1 and PR-AUC rise mechanically with prevalence.")
    a("- **Template semantics**: BGL alerts are often intrinsically-rare hardware/kernel "
      "templates that a normal-only model has literally never seen — an easier surprise "
      "signal than HDFS's ordering anomalies among common events.")
    a("- **Synthetic sessions**: fixed 100-line windows in time order mean temporally-"
      "adjacent, near-duplicate windows can straddle a random split. The chronological-"
      "split check below quantifies exactly this optimism.")
    if chrono:
        a(f"- **Chronological split**: detection F1 goes from {bdet['f1']:.3f} (random) to "
          f"**{chrono['f1']:.3f}** (train on first 70%, test on last 20%), PR-AUC "
          f"{bdet['pr_auc']:.3f} → {chrono['pr_auc']:.3f}. The gap measures how much the "
          "random split flatters BGL; the chrono number is the deployment-realistic one.")
    a("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L) + "\n")
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
def _evaluate_dataset(name, device, skip_llm, llm_n):
    cfg = config_for_dataset(name)
    encoder = StreamingLabelEncoder.load(os.path.join(cfg["cache_dir"], "encoder.pkl"))
    event_map = load_event_templates(cfg["datasets"][name]["templates"])
    ckpt = cfg["detection_checkpoint"]
    flags = variant_flags("gru")
    model = GRUAnomalyDetector(
        len(encoder.classes_), cfg["embedding_dim"], cfg["hidden_dim"],
        cfg["num_layers"], cfg["dropout"], **flags).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log.info("[%s] detector: %s (vocab %d)", name, ckpt, len(encoder.classes_))

    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, cfg)
    tr_idx, va_idx, te_idx = bd.split_block_indices(labels, cfg)
    tr_blocks, tr_y, _ = bd.select_blocks(blocks, labels, block_ids, tr_idx)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, _ = bd.select_blocks(blocks, labels, block_ids, te_idx)
    n_train_normal = int((np.asarray(tr_y) == 0).sum())

    det = detection_eval(model, va_blocks, np.asarray(va_y), te_blocks, np.asarray(te_y),
                         device, cfg)
    log.info("[%s] detection F1 %.4f  PR-AUC %.4f  AUROC %.4f  (%.0f blocks/s)",
             name, det["f1"], det["pr_auc"], det["auroc"], det["blocks_per_s"])
    nxt = next_event_metrics(model, te_blocks, device, cfg)
    log.info("[%s] next-event Top-1 %.4f  wF1 %.4f  macroF1 %.4f",
             name, nxt["top1"], nxt["weighted_f1"], nxt["macro_f1"])

    llm = None
    if not skip_llm:
        llm = measure_llm_latency(model, te_blocks, det, device, cfg, encoder, event_map, n=llm_n)
        if llm:
            log.info("[%s] Llama latency %.2f s/call (n=%d)", name, llm["avg_s"], llm["n"])

    meta = {"vocab": len(encoder.classes_), "n_blocks": len(blocks),
            "n_train": len(tr_blocks), "n_train_normal": n_train_normal,
            "n_val": len(va_blocks), "n_test": len(te_blocks),
            "test_rate": float(np.mean(te_y))}
    out = {"checkpoint": ckpt, "meta": meta, "next_event": nxt, "detection": det, "llm": llm}
    # keep the raw blocks around only for BGL's chrono check
    return out, (blocks, labels, encoder, cfg)


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Measured BGL evaluation + HDFS comparison.")
    ap.add_argument("--skip-chrono", action="store_true",
                    help="Skip the chronological-split retraining check (~15-20 min).")
    ap.add_argument("--skip-llm", action="store_true", help="Skip Llama latency measurement.")
    ap.add_argument("--chrono-epochs", type=int, default=8)
    ap.add_argument("--chrono-patience", type=int, default=3)
    ap.add_argument("--llm-n", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(CONFIG["random_seed"]); np.random.seed(CONFIG["random_seed"])
    device = get_device()
    log.info("Device: %s", device)

    results = {"device": str(device)}
    results["bgl"], bgl_raw = _evaluate_dataset("bgl", device, args.skip_llm, args.llm_n)
    results["hdfs"], _ = _evaluate_dataset("hdfs", device, args.skip_llm, args.llm_n)

    chrono = None
    if not args.skip_chrono:
        blocks, labels, encoder, cfg = bgl_raw
        log.info("Chronological-split robustness check (BGL) ...")
        chrono = chrono_split_eval(blocks, labels, encoder, device, cfg,
                                   args.chrono_epochs, args.chrono_patience)
        log.info("Chrono: F1 %.4f  PR-AUC %.4f  AUROC %.4f  (random: F1 %.4f)",
                 chrono["f1"], chrono["pr_auc"], chrono["auroc"],
                 results["bgl"]["detection"]["f1"])

    results["peak_rss_mb"] = peak_rss_mb()
    costs = {
        "bgl": parse_training_costs("outputs/bgl/normal_only_training_report.md"),
        "hdfs": parse_training_costs(".checkpoints/timing_hdfs/normal_only_training_report.md"),
    }
    write_bgl_results(results, chrono, costs)
    write_bgl_vs_hdfs(results, chrono, costs)
    log.info("Done. ✓")


if __name__ == "__main__":
    main()

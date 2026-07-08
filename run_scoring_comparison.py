# =============================================================================
# run_scoring_comparison.py — Tasks 3 & 4: which anomaly SCORE + AGGREGATION best
# separates anomalous blocks from normal ones?
#
# Holds the model fixed (the normal-only detector by default) and varies ONLY the
# scoring rule, on the honest held-out TEST block split. The F1 threshold is chosen
# on VALIDATION and applied to TEST (no oracle optimism). All scores are computed
# from ONE inference pass over per-window probabilities.
#
# Per-window signals:            Block aggregations (Task 4):
#   * p_true   = P(true next)      * max  — the single most surprising transition
#   * 1 - p    = surprisal (cur.)  * mean — average surprise over the block
#   * -log p   = surprise (NLL)    * sum/count — DeepLog-style violation count
#   * 1 - MSP  = self-uncertainty  * rolling — max of a length-W moving average
#   * top-k miss (true ∉ top-k)      (a sustained burst, not one spike)
#   * combined — z(max NLL) + z(top-k-miss count), z-fit on VAL (no leakage)
#
# Output: outputs/anomaly_scoring_comparison.md (AUROC, PR-AUC, F1 per score).
#
# Usage (smoke, CPU): venv/bin/python run_scoring_comparison.py --max-blocks 20000 --device cpu
# Usage (full):       venv/bin/python run_scoring_comparison.py
# =============================================================================

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from config import CONFIG, get_device, variant_flags, config_for_dataset
from utils.logging_utils import setup_logging
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from evaluation.anomaly_eval import _metrics_at_best_f1

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One inference pass → per-window signals (mirrors anomaly_eval.score_blocks'
# bucketed batching, but keeps the raw per-window values so every score can be
# derived without re-running the model).
# ---------------------------------------------------------------------------
def per_window_signals(model, blocks, device, config):
    seq_len = config["sequence_length"]
    k = config.get("topk_k", 3)
    batch = config["batch_size"]

    win_block, win_ctx, win_tgt, win_len = [], [], [], []
    for bi, blk in enumerate(blocks):
        for i in range(1, len(blk)):
            ctx = blk[max(0, i - seq_len):i]
            win_block.append(bi); win_ctx.append(ctx)
            win_tgt.append(int(blk[i])); win_len.append(len(ctx))
    win_block = np.array(win_block); win_tgt = np.array(win_tgt); win_len = np.array(win_len)
    n = len(win_tgt)

    p_true = np.zeros(n, np.float32)
    msp = np.zeros(n, np.float32)
    miss = np.zeros(n, np.float32)

    model.eval()
    with torch.no_grad():
        for length in np.unique(win_len):
            rows = np.where(win_len == length)[0]
            ctx_mat = np.stack([win_ctx[r] for r in rows])
            tgt = win_tgt[rows]
            for s in range(0, len(rows), batch):
                bx = torch.from_numpy(ctx_mat[s:s + batch]).long().to(device)
                bt = tgt[s:s + batch]
                probs = torch.softmax(model(bx), dim=1).cpu().numpy()
                idx = np.arange(len(bt))
                p_true[rows[s:s + batch]] = probs[idx, bt]
                msp[rows[s:s + batch]] = probs.max(axis=1)
                topk = np.argpartition(probs, -k, axis=1)[:, -k:]
                miss[rows[s:s + batch]] = (~(topk == bt[:, None]).any(axis=1)).astype(np.float32)

    nll = -np.log(np.clip(p_true, 1e-12, 1.0))
    return {"win_block": win_block, "win_len": win_len, "win_tgt": win_tgt,
            "n_blocks": len(blocks),
            "p_true": p_true, "nll": nll, "msp": msp, "miss": miss}


def _agg(arr, win_block, n_blocks, how):
    out = np.zeros(n_blocks, np.float32)
    if how == "max":
        np.maximum.at(out, win_block, arr)
    elif how == "sum":
        np.add.at(out, win_block, arr)
    elif how == "mean":
        cnt = np.zeros(n_blocks, np.float32)
        np.add.at(out, win_block, arr); np.add.at(cnt, win_block, 1.0)
        out = out / np.maximum(cnt, 1.0)
    return out


def _rolling_max(sig, value_key, window):
    """Per block: max over a length-`window` moving average of the per-window value
    (captures a sustained BURST of surprise rather than one spike)."""
    nb = sig["n_blocks"]; wb = sig["win_block"]; vals = sig[value_key]
    out = np.zeros(nb, np.float32)
    # group window rows by block (rows are already in block order from construction)
    order = np.argsort(wb, kind="stable")
    wb_s = wb[order]; v_s = vals[order]
    starts = np.searchsorted(wb_s, np.arange(nb), side="left")
    ends = np.searchsorted(wb_s, np.arange(nb), side="right")
    for b in range(nb):
        v = v_s[starts[b]:ends[b]]
        if len(v) == 0:
            continue
        if len(v) < window:
            out[b] = float(v.mean())
        else:
            csum = np.cumsum(np.insert(v, 0, 0.0))
            means = (csum[window:] - csum[:-window]) / window
            out[b] = float(means.max())
    return out


def build_scores(sig, ref_sig=None, roll_window=5):
    """All candidate block scores. `ref_sig` (VAL) is used to z-fit the combined
    score so test never sees its own normalisation statistics."""
    wb, nb = sig["win_block"], sig["n_blocks"]
    s = {}
    s["msp_self_uncertainty (1-MSP, max)"] = _agg(1.0 - sig["msp"], wb, nb, "max")
    s["surprisal_1mp_max"] = _agg(1.0 - sig["p_true"], wb, nb, "max")
    s["surprisal_1mp_mean"] = _agg(1.0 - sig["p_true"], wb, nb, "mean")
    s["nll_-logp_max"] = _agg(sig["nll"], wb, nb, "max")
    s["nll_-logp_mean"] = _agg(sig["nll"], wb, nb, "mean")
    s["topk_miss_count"] = _agg(sig["miss"], wb, nb, "sum")
    s["topk_miss_frac"] = _agg(sig["miss"], wb, nb, "mean")
    s[f"nll_rolling_max(w={roll_window})"] = _rolling_max(sig, "nll", roll_window)

    # Combined = z(max NLL) + z(top-k-miss count); z stats fit on the reference
    # split (VAL) to avoid using test's own distribution.
    base = ref_sig if ref_sig is not None else sig
    rb, rnb = base["win_block"], base["n_blocks"]
    ref_nll_max = _agg(base["nll"], rb, rnb, "max")
    ref_miss_cnt = _agg(base["miss"], rb, rnb, "sum")

    def z(x, ref):
        mu, sd = float(ref.mean()), float(ref.std() + 1e-8)
        return (x - mu) / sd
    s["combined_z(nll_max)+z(miss_count)"] = (
        z(s["nll_-logp_max"], ref_nll_max) + z(s["topk_miss_count"], ref_miss_cnt))

    # Per-EVENT-TYPE calibration (ROADMAP item 1; aimed at large-vocab datasets
    # like BGL where the baseline NLL of a window varies wildly by which template
    # is the target). Each window's NLL is z-scored against ITS TARGET EVENT's
    # NLL distribution on the reference split, so "surprising for THIS event"
    # replaces "surprising in general". Events with <5 reference windows fall
    # back to the global stats (their per-event estimates are unstable).
    if "win_tgt" in sig and "win_tgt" in base:
        vocab_hi = int(max(sig["win_tgt"].max(), base["win_tgt"].max())) + 1
        cnt = np.zeros(vocab_hi); tot = np.zeros(vocab_hi); sq = np.zeros(vocab_hi)
        np.add.at(cnt, base["win_tgt"], 1.0)
        np.add.at(tot, base["win_tgt"], base["nll"])
        np.add.at(sq, base["win_tgt"], base["nll"].astype(np.float64) ** 2)
        g_mu = float(base["nll"].mean()); g_sd = float(base["nll"].std() + 1e-8)
        mu = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), g_mu)
        var = sq / np.maximum(cnt, 1.0) - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-8))
        low = cnt < 5
        mu[low], sd[low] = g_mu, g_sd
        wz = ((sig["nll"] - mu[sig["win_tgt"]]) / sd[sig["win_tgt"]]).astype(np.float32)
        s["nll_perevent_z_mean"] = _agg(wz, wb, nb, "mean")
        s["nll_perevent_z_max"] = _agg(wz, wb, nb, "max")
    return s


def band_sizes(val_score, val_y, max_normal_rate=0.01):
    """Normal/Uncertain/Suspicious counts on VAL for one candidate score. Mirrors
    pipeline.detection_pipeline.derive_bands's threshold logic (duplicated, not
    imported — that module imports FROM this one, so importing it back here would
    be circular). Used to check whether a candidate that wins on PR-AUC still
    leaves a WORKABLE Uncertain band: the routing architecture sends Uncertain
    blocks to Llama for adjudication, so a score that separates so sharply it
    leaves zero blocks in the middle defeats that stage even though it detects
    better in isolation."""
    suspicious_thr = _metrics_at_best_f1(val_y, val_score)["threshold"]
    order = np.argsort(val_score, kind="stable")
    ys = val_y[order].astype(np.float64)
    ss = val_score[order]
    cum_rate = np.cumsum(ys) / (np.arange(len(ys)) + 1.0)
    ok = np.where(cum_rate <= max_normal_rate)[0]
    if len(ok):
        k = int(ok.max())
        uncertain_thr = float(ss[min(k + 1, len(ss) - 1)])
    else:
        uncertain_thr = float(ss[0])
    uncertain_thr = min(uncertain_thr, suspicious_thr)
    n_normal = int((val_score < uncertain_thr).sum())
    n_uncertain = int(((val_score >= uncertain_thr) & (val_score < suspicious_thr)).sum())
    n_suspicious = int((val_score >= suspicious_thr).sum())
    return n_normal, n_uncertain, n_suspicious


def evaluate_scores(val_scores, val_y, test_scores, test_y):
    rows = {}
    for key in test_scores:
        thr = _metrics_at_best_f1(val_y, val_scores[key])["threshold"]   # picked on VAL
        y_pred = (test_scores[key] >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            test_y, y_pred, average="binary", zero_division=0)
        rows[key] = {
            "auroc": float(roc_auc_score(test_y, test_scores[key])),
            "pr_auc": float(average_precision_score(test_y, test_scores[key])),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "val_pr_auc": float(average_precision_score(val_y, val_scores[key])),
        }
    return rows


def write_report(rows, meta, model_tag, out_dir="outputs"):
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["val_pr_auc"], reverse=True)
    best = ranked[0][0]
    L = []; a = L.append
    a("# Anomaly-Scoring Comparison (Tasks 3 & 4)\n")
    a(f"_Model held fixed (**{model_tag}**); only the scoring rule varies. Honest "
      f"held-out TEST block split; the F1 threshold is chosen on VALIDATION and "
      f"applied to TEST. Best score chosen by validation PR-AUC._\n")
    a(f"- Test blocks: **{meta['n_test']:,}** · anomaly base rate **{meta['test_rate']:.3f}** "
      f"· val blocks **{meta['n_val']:,}**\n")
    a(f"**Best separator (by val PR-AUC): `{best}`**\n")
    a("| Score (signal + aggregation) | AUROC | PR-AUC | Precision | Recall | F1 |")
    a("|---|---|---|---|---|---|")
    for key, m in ranked:
        star = " ⭐" if key == best else ""
        a(f"| {key}{star} | {m['auroc']:.4f} | **{m['pr_auc']:.4f}** | "
          f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    a("")
    a("## Reading this\n")
    a("- **Signal:** `-log p` (NLL) is the proper surprise — its unbounded tail "
      "separates a p≈1e-6 transition from a merely-unlikely one, unlike `1 - p` which "
      "saturates at 1. **MSP** is the model's confidence in its OWN top event "
      "(the hybrid-pipeline signal); `1 - MSP` is its self-uncertainty.")
    a("- **Aggregation (Task 4):** `max` flags the single most surprising transition "
      "(DeepLog's rule); `count` is the number of out-of-top-k violations; `mean` is "
      "the average surprise; `rolling_max` is the most surprising sustained BURST.")
    a("- **Combined** standardises and sums the two strongest orthogonal signals "
      "(peak surprise + violation count), z-fit on validation.\n")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "anomaly_scoring_comparison.md"
    path.write_text("\n".join(L) + "\n")
    log.info("Wrote %s", path)


# Qualitative notes per scoring rule (signal + aggregation properties). Used to turn
# the raw ranking into the strengths/weaknesses section the audit asked for (Task 2).
_SCORE_NOTES = {
    "nll_-logp_mean": (
        "Proper unbounded surprise (−log p) averaged over the block — well-calibrated, "
        "robust to block length.",
        "A single catastrophic transition can be diluted by many ordinary ones in a long block."),
    "nll_-logp_max": (
        "Flags the single most surprising transition (DeepLog's rule) on the unbounded NLL scale.",
        "One noisy spike (a rare-but-benign event) can trigger a false positive."),
    "surprisal_1mp_max": (
        "DeepLog max rule on a bounded [0,1] scale — directly interpretable as a probability.",
        "`1 − p` saturates at 1, so it cannot tell a p≈1e-3 transition from a p≈1e-9 one."),
    "surprisal_1mp_mean": (
        "Smooth average surprise, bounded [0,1].",
        "Saturates like 1−p and dilutes single spikes."),
    "topk_miss_frac": (
        "Classic DeepLog top-k violation rate; parameter-light and interpretable.",
        "Binary per window → coarse; near-random unless the model is trained normal-only."),
    "topk_miss_count": (
        "Raw count of out-of-top-k violations in the block.",
        "Confounded with block length; coarse, low PR-AUC."),
    "combined_z(nll_max)+z(miss_count)": (
        "Fuses two orthogonal signals (peak surprise + violation count), z-fit on validation.",
        "Adds a moving part; only helps when BOTH component signals are informative."),
    "msp_self_uncertainty (1-MSP, max)": (
        "The model's confidence in its OWN top prediction — the best 'is this prediction correct' signal.",
        "Measures self-confidence, NOT whether the ACTUAL event was surprising → the WORST block detector."),
    "nll_perevent_z_mean": (
        "Per-event-type calibration: each window's NLL is z-scored against its TARGET event's "
        "reference distribution — 'surprising for THIS template', key on large vocabularies.",
        "Needs enough reference windows per event; rare templates fall back to global stats."),
    "nll_perevent_z_max": (
        "Per-event-calibrated peak surprise (DeepLog max rule after per-template z-scoring).",
        "A single noisy spike on a poorly-estimated template can trigger a false positive."),
}


def _rolling_note(key):
    return ("Captures a sustained BURST of surprise (max over a moving average), not one spike — "
            "robust to isolated noise.",
            "Window size is a hyperparameter; weak on very short blocks.")


def write_detection_analysis(rows, meta, model_tag, out_dir="outputs",
                             val_scores=None, val_y=None, max_normal_rate=0.01):
    """Task 2 deliverable: outputs/detection_analysis.md — ranking + per-score
    strengths/weaknesses + a recommendation, on top of the raw comparison table.

    `val_scores`/`val_y` (optional) let the Gate check also verify the candidate
    doesn't collapse the Uncertain band (see band_sizes' docstring) — a score can
    win on PR-AUC while defeating the routing architecture's need for a workable
    'send to Llama' middle band."""
    by_val = sorted(rows.items(), key=lambda kv: kv[1]["val_pr_auc"], reverse=True)
    by_f1 = sorted(rows.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    best = by_val[0][0]
    best_f1 = by_f1[0][0]
    worst = by_val[-1][0]

    L = []; a = L.append
    a("# Detection-Score Analysis — which surprise signal best finds suspicious blocks (Task 2)\n")
    a(f"_Model held fixed (**{model_tag}**, the normal-only detector); only the scoring rule "
      "varies. Honest held-out TEST block split; the operating threshold is chosen on "
      "VALIDATION and applied to TEST. Primary ranking metric: **validation PR-AUC** "
      "(anomalies are rare, so PR-AUC matters more than AUROC). Generated by "
      "`run_scoring_comparison.py`._\n")
    a(f"- Test blocks: **{meta['n_test']:,}** · val blocks **{meta['n_val']:,}** · "
      f"anomaly base rate **{meta['test_rate']:.3f}**\n")

    a("## 1. Ranking (by validation PR-AUC)\n")
    a("| # | Score (signal + aggregation) | Val PR-AUC | Test AUROC | Test PR-AUC | "
      "Test Precision | Test Recall | Test F1 |")
    a("|---|---|---|---|---|---|---|---|")
    for i, (key, m) in enumerate(by_val, 1):
        star = " ⭐" if key == best else ""
        a(f"| {i} | {key}{star} | **{m['val_pr_auc']:.4f}** | {m['auroc']:.4f} | "
          f"{m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    a("")

    a("## 2. Strengths & weaknesses\n")
    a("| Score | Strength | Weakness |")
    a("|---|---|---|")
    for key, _ in by_val:
        s, w = _rolling_note(key) if key.startswith("nll_rolling_max") else _SCORE_NOTES.get(
            key, ("—", "—"))
        a(f"| {key} | {s} | {w} |")
    a("")

    a("## 3. Recommendation\n")
    a(f"- **Primary detector signal: `{best}`** — the best separator by validation PR-AUC "
      f"(val PR-AUC {rows[best]['val_pr_auc']:.4f}; test PR-AUC {rows[best]['pr_auc']:.4f}, "
      f"F1 {rows[best]['f1']:.4f}). This is what the production detection pipeline should use "
      "to rank blocks.")
    default_key = "nll_-logp_mean"
    if best != default_key and default_key in rows:
        gap = rows[best]["val_pr_auc"] - rows[default_key]["val_pr_auc"]
        meets_gap = gap >= 0.005
        band_note = ""
        adopt = meets_gap
        if meets_gap and val_scores is not None and val_y is not None:
            n_v, n_best_unc, n_v2 = band_sizes(val_scores[best], val_y, max_normal_rate)
            n_d, n_def_unc, n_d2 = band_sizes(val_scores[default_key], val_y, max_normal_rate)
            n_val = len(val_y)
            min_unc = max(20, int(0.01 * n_val))   # workable Llama-adjudication sample
            viable = n_best_unc >= min_unc
            adopt = meets_gap and viable
            band_note = (
                f" Uncertain-band check (val, min workable size {min_unc:,}): `{best}` leaves "
                f"**{n_best_unc:,}** blocks Uncertain vs `{default_key}`'s {n_def_unc:,}. "
                + ("This is workable for Llama adjudication." if viable else
                   f"**This is below the workable minimum — adopting `{best}` would starve the "
                   "Uncertain→Llama adjudication stage of cases, defeating the routing "
                   "architecture even though it detects better in isolation.**"))
        a(f"- **Gate check vs the current production default (`{default_key}`):** val PR-AUC "
          f"gap = {gap:+.4f} ({rows[default_key]['val_pr_auc']:.4f} → "
          f"{rows[best]['val_pr_auc']:.4f}), {'meets' if meets_gap else 'does not meet'} the "
          f"≥0.005 bar.{band_note}")
        a(f"- **Decision: {'Adopt' if adopt else 'Do NOT adopt'} `{best}` as the production "
          f"detection score for this dataset.**")
    if best_f1 != best:
        a(f"- Highest raw test-F1 at its val-selected threshold is `{best_f1}` "
          f"(F1 {rows[best_f1]['f1']:.4f}); it is a reasonable alternative if recall is the priority, "
          "but the headline uses the PR-AUC winner to avoid threshold-luck.")
    worst_weakness = (_rolling_note(worst) if worst.startswith("nll_rolling_max")
                      else _SCORE_NOTES.get(worst, ("—", "—")))[1]
    a(f"- **Key finding — do NOT use `{worst}` for detection here.** It is the worst separator "
      f"on this dataset (val PR-AUC {rows[worst]['val_pr_auc']:.4f}, AUROC {rows[worst]['auroc']:.3f}). "
      f"{worst_weakness}")
    msp_key = "msp_self_uncertainty (1-MSP, max)"
    if worst != msp_key and msp_key in rows:
        a(f"- **MSP self-uncertainty is also unsuitable as a detector** (val PR-AUC "
          f"{rows[msp_key]['val_pr_auc']:.4f} here) even though it is not the worst on this "
          "dataset: it measures the model's confidence in its OWN top prediction (great for "
          "routing 'is this prediction correct?'), NOT whether the ACTUAL event was surprising — "
          "what suspicious-activity detection needs. The old hybrid pipeline routes correctness; "
          "the detector must route surprise.")
    a("- `-log p` (NLL) beats `1 − p`: NLL's unbounded tail distinguishes a p≈1e-6 transition "
      "from a merely-unlikely one, while `1 − p` saturates at 1.\n")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "detection_analysis.md"
    path.write_text("\n".join(L) + "\n")
    log.info("Wrote %s", path)


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Anomaly-scoring comparison (Tasks 3 & 4).")
    ap.add_argument("--checkpoint", default=None,
                    help="Model to score with (default: the active dataset's "
                         "detection_checkpoint, i.e. its normal-only detector).")
    ap.add_argument("--tag", default=None, help="Label for the report.")
    ap.add_argument("--max-blocks", type=int, default=None, help="Random subsample for speed.")
    ap.add_argument("--device", default=None, help="Override device (e.g. cpu).")
    ap.add_argument("--roll-window", type=int, default=5)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name from CONFIG['datasets'] (default: CONFIG['dataset'] = 'hdfs').")
    ap.add_argument("--out-dir", default=None,
                    help="Report directory (default: the dataset's output_dir, or outputs/).")
    args = ap.parse_args()

    cfg = config_for_dataset(args.dataset)
    checkpoint = args.checkpoint or cfg["detection_checkpoint"]
    tag = args.tag or f"{cfg['dataset']}_normal_only"
    out_dir = args.out_dir or cfg.get("output_dir", "outputs")

    device = torch.device(args.device) if args.device else get_device()
    log.info("Device: %s | dataset: %s | checkpoint: %s", device, cfg["dataset"], checkpoint)

    encoder = StreamingLabelEncoder.load(os.path.join(cfg["cache_dir"], "encoder.pkl"))
    flags = variant_flags("gru")
    model = GRUAnomalyDetector(
        len(encoder.classes_), cfg["embedding_dim"], cfg["hidden_dim"],
        cfg["num_layers"], cfg["dropout"], **flags).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, cfg, max_blocks=args.max_blocks)
    _, va_idx, te_idx = bd.split_block_indices(labels, cfg)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, _ = bd.select_blocks(blocks, labels, block_ids, te_idx)

    log.info("Scoring %d val + %d test blocks ...", len(va_blocks), len(te_blocks))
    val_sig = per_window_signals(model, va_blocks, device, cfg)
    test_sig = per_window_signals(model, te_blocks, device, cfg)

    val_scores = build_scores(val_sig, ref_sig=val_sig, roll_window=args.roll_window)
    test_scores = build_scores(test_sig, ref_sig=val_sig, roll_window=args.roll_window)
    rows = evaluate_scores(val_scores, va_y, test_scores, te_y)

    meta = {"n_test": len(te_blocks), "n_val": len(va_blocks), "test_rate": float(np.mean(te_y))}
    write_report(rows, meta, tag, out_dir=out_dir)
    write_detection_analysis(rows, meta, tag, out_dir=out_dir, val_scores=val_scores, val_y=va_y,
                             max_normal_rate=cfg.get("detection_max_normal_rate", 0.01))

    ranked = sorted(rows.items(), key=lambda kv: kv[1]["val_pr_auc"], reverse=True)
    print("\n" + "=" * 70)
    print(f"  ANOMALY-SCORING COMPARISON  (model: {tag}; held-out test)")
    print("=" * 70)
    for key, m in ranked:
        print(f"  {key:38s}  AUROC {m['auroc']:.3f}  PR-AUC {m['pr_auc']:.3f}  F1 {m['f1']:.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

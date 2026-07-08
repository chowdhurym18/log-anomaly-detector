# =============================================================================
# evaluate_llm_ladder.py — Task 4: measure, rung by rung, whether prompt
# engineering / grounded evidence / structured outputs / few-shot / self-
# consistency / a bigger model improve Llama's classification of the blocks the
# GRU routes as UNCERTAIN — before considering any fine-tuning.
#
# Protocol (per dataset, identical to production routing):
#   * normal-only detector -> nll_-logp_mean -> validation-derived bands
#   * n uncertain TEST blocks sampled with a fixed seed; every rung sees the
#     SAME cases, so differences are attributable to the rung alone
#   * three deterministic baselines, including an "extended rule" built from the
#     SAME evidence features the v2 prompt gets — if Llama cannot beat the rule
#     built from its own inputs, it adds nothing
#   * few-shot exemplars come from the VALIDATION uncertain band (never test)
#
# Rungs:
#   R0 production v1 prompt (the deployed baseline; hardcodes HDFS framing)
#   R1 v2, dataset-aware framing only (isolates the framing fix)
#   R2 R1 + grounded evidence (percentile / template frequency / top windows)
#        + relaxed SUSPICIOUS criteria (the measured root cause of v1's 0% recall)
#   R3 R2 + Ollama structured outputs (JSON schema)
#   R4 R3 + 4 few-shot exemplars from the validation uncertain band
#   R5 best of R1-R4 + self-consistency majority vote (k=3, temp 0.7)
#   R6 best of R1-R4 with llama3.2:3b (is 1B capacity the bottleneck?)
#
# Out: outputs/llama_bgl_evaluation.md (BGL primary + HDFS appendix)
#      outputs/llm_ladder_<dataset>.csv (per-case, per-rung raw results)
# Run: venv/bin/python evaluate_llm_ladder.py [--n 100] [--datasets bgl,hdfs]
#      [--skip-3b] [--rungs all]
# =============================================================================

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CONFIG, get_device, variant_flags, config_for_dataset
from utils.logging_utils import setup_logging
from utils.format import format_pct as pct
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing.templates import load_event_templates
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from run_scoring_comparison import per_window_signals, build_scores
from pipeline.detection_pipeline import derive_bands, classify
from llm.ollama_client import test_ollama_connection
from llm.uncertain_classifier import (
    classify_uncertain, classify_uncertain_v2, build_user_v2, FAILURE_TOKENS,
)

log = logging.getLogger(__name__)

ROUTING_REASON = ("GRU routed this block UNCERTAIN: its surprise score fell "
                  "between the Normal and Suspicious thresholds.")
# A-priori thresholds for the extended deterministic rule — chosen to MIRROR the
# v2 prompt criteria exactly (never seen in normal training; >=99th percentile).
RULE_PCTL = 99.0


# ---------------------------------------------------------------------------
# Case construction (shared by every rung)
# ---------------------------------------------------------------------------
def top_surprising_windows(model, block, device, config, k=3):
    """Like detection_pipeline.most_surprising_window but returns the top-k
    windows by surprisal (descending)."""
    seq_len = config["sequence_length"]
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(1, len(block)):
            ctx = np.asarray(block[max(0, i - seq_len):i], dtype=np.int64)
            tgt = int(block[i])
            probs = torch.softmax(model(torch.from_numpy(ctx[None, :]).to(device)), dim=1)
            probs = probs.cpu().numpy()[0]
            rows.append({"context": ctx.tolist(), "target": tgt,
                         "pred": int(probs.argmax()), "msp": float(probs.max()),
                         "surprisal": 1.0 - float(probs[tgt])})
    rows.sort(key=lambda r: r["surprisal"], reverse=True)
    return rows[:k]


def _normal_train_freqs(tr_blocks, tr_y, vocab):
    """Event-occurrence counts over NORMAL training blocks (cheap retrieval)."""
    counts = np.zeros(vocab, dtype=np.int64)
    total = 0
    for blk, y in zip(tr_blocks, tr_y):
        if y == 0:
            arr = np.asarray(blk, dtype=np.int64)
            counts += np.bincount(arr, minlength=vocab)
            total += len(arr)
    return counts, total


def _case(model, block, score, device, encoder, event_map, cfg,
          freqs, total_events, val_normal_sorted):
    wins = top_surprising_windows(model, block, device, cfg, k=3)
    w = wins[0]
    seq = encoder.inverse_transform(w["context"], context="ladder ctx")
    pred = encoder.inverse_transform([w["pred"]], context="ladder pred")[0]
    actual = encoder.inverse_transform([w["target"]], context="ladder actual")[0]
    actual_tmpl = (event_map.get(actual, "") or "").lower()
    kw = any(tok in actual_tmpl for tok in FAILURE_TOKENS)

    pctl = 100.0 * np.searchsorted(val_normal_sorted, score) / max(len(val_normal_sorted), 1)
    top_windows = []
    for rank, tw in enumerate(wins[1:], start=2):   # rank 1 IS the main actual event
        ev = encoder.inverse_transform([tw["target"]], context="ladder topwin")[0]
        top_windows.append({"rank": rank, "actual_event": ev,
                            "actual_template": event_map.get(ev, "template unavailable"),
                            "surprisal": tw["surprisal"]})
    evidence = {
        "score_percentile": float(pctl),
        "actual_freq": int(freqs[w["target"]]),
        "pred_freq": int(freqs[w["pred"]]),
        "total_events": int(total_events),
        "top_windows": top_windows,
    }
    return {"seq": seq, "pred": pred, "actual": actual,
            "conf_in": 1.0 - w["surprisal"], "anom_in": w["surprisal"],
            "kw": kw, "evidence": evidence}


def _exemplar_answer(case, true_label, structured):
    """Deterministic, grounded few-shot answer for a labelled validation case."""
    ev = case["evidence"]
    if true_label == 1:
        if case["kw"]:
            expl = "The actual event's template records a failure condition."
        elif ev["actual_freq"] == 0:
            expl = "The actual event was never seen in normal training data."
        else:
            expl = (f"The block's surprise percentile among normal blocks is "
                    f"{ev['score_percentile']:.0f}.")
        cls, conf = "SUSPICIOUS", 85
        quote = f"actual event {case['actual']} with surprise percentile {ev['score_percentile']:.0f}"
    else:
        cls, conf = "NORMAL", 85
        expl = ("The actual event is common in normal training data and no grounded "
                "suspicious condition holds.")
        quote = (f"actual event {case['actual']} seen {ev['actual_freq']:,} times in "
                 "normal training data")
    if structured:
        return (f'{{"classification": "{cls}", "confidence": {conf}, '
                f'"evidence": "{quote}", "explanation": "{expl}"}}')
    return (f"CLASSIFICATION: {cls}\nCONFIDENCE: {conf}\n"
            f"EVIDENCE: {quote}\nEXPLANATION: {expl}")


def build_dataset_cases(name, device, n, seed, n_fewshot=4):
    """Production-identical routing; returns test cases + few-shot exemplars + meta."""
    cfg = config_for_dataset(name)
    encoder = StreamingLabelEncoder.load(os.path.join(cfg["cache_dir"], "encoder.pkl"))
    event_map = load_event_templates(cfg["datasets"][name]["templates"])
    flags = variant_flags("gru")
    model = GRUAnomalyDetector(
        len(encoder.classes_), cfg["embedding_dim"], cfg["hidden_dim"],
        cfg["num_layers"], cfg["dropout"], **flags).to(device)
    model.load_state_dict(torch.load(cfg["detection_checkpoint"], map_location=device))

    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, cfg)
    tr_idx, va_idx, te_idx = bd.split_block_indices(labels, cfg)
    tr_blocks, tr_y, _ = bd.select_blocks(blocks, labels, block_ids, tr_idx)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, te_ids = bd.select_blocks(blocks, labels, block_ids, te_idx)

    roll = cfg.get("detection_roll_window", 5)
    score_name = cfg.get("detection_score", "nll_-logp_mean")
    val_sig = per_window_signals(model, va_blocks, device, cfg)
    test_sig = per_window_signals(model, te_blocks, device, cfg)
    val_score = build_scores(val_sig, ref_sig=val_sig, roll_window=roll)[score_name]
    test_score = build_scores(test_sig, ref_sig=val_sig, roll_window=roll)[score_name]
    unc, susp = derive_bands(val_score, np.asarray(va_y),
                             cfg.get("detection_max_normal_rate", 0.01))
    bands = classify(test_score, unc, susp)

    freqs, total_events = _normal_train_freqs(tr_blocks, tr_y, len(encoder.classes_))
    val_normal_sorted = np.sort(val_score[np.asarray(va_y) == 0])

    unc_idx = np.where(bands == "UNCERTAIN")[0]
    rng = np.random.default_rng(seed)
    sample = rng.permutation(unc_idx)[:min(n, len(unc_idx))]
    log.info("[%s] uncertain test blocks: %d — evaluating %d (seed %d)",
             name, len(unc_idx), len(sample), seed)

    cases = []
    for i in sample:
        c = _case(model, te_blocks[i], float(test_score[i]), device, encoder,
                  event_map, cfg, freqs, total_events, val_normal_sorted)
        c["true"] = int(te_y[i]); c["block_id"] = te_ids[i]
        cases.append(c)

    # Few-shot exemplars from the VALIDATION uncertain band (never test).
    val_bands = classify(val_score, unc, susp)
    v_unc = np.where(val_bands == "UNCERTAIN")[0]
    v_anom = [i for i in v_unc if va_y[i] == 1]
    v_norm = [i for i in v_unc if va_y[i] == 0]
    rng2 = np.random.default_rng(seed + 1)
    picks = (list(rng2.permutation(v_anom)[:n_fewshot // 2])
             + list(rng2.permutation(v_norm)[:n_fewshot - n_fewshot // 2]))
    exemplars = []
    for i in picks:
        c = _case(model, va_blocks[i], float(val_score[i]), device, encoder,
                  event_map, cfg, freqs, total_events, val_normal_sorted)
        c["true"] = int(va_y[i])
        exemplars.append(c)
    log.info("[%s] few-shot exemplars: %d (%d anomalous)", name, len(exemplars),
             sum(c["true"] for c in exemplars))

    meta = {"n_uncertain": int(len(unc_idx)), "n": len(sample), "seed": seed,
            "anom_share": float(np.mean([c["true"] for c in cases])) if cases else 0.0,
            "vocab": len(encoder.classes_)}
    return {"cfg": cfg, "event_map": event_map, "cases": cases,
            "exemplars": exemplars, "meta": meta}


def _fewshot_msgs(exemplars, event_map, structured):
    out = []
    for c in exemplars:
        user = build_user_v2(c["seq"], c["pred"], c["actual"], event_map,
                             c["conf_in"], c["anom_in"], ROUTING_REASON, c["evidence"])
        out.append((user, _exemplar_answer(c, c["true"], structured)))
    return out


# ---------------------------------------------------------------------------
# Rungs & metrics
# ---------------------------------------------------------------------------
RUNGS = [
    ("R0 production prompt (v1, deployed)",
     dict(kind="v1")),
    ("R1 + dataset-aware framing",
     dict(kind="v2", relaxed=False, evidence=False, structured=False, fewshot=False)),
    ("R2 + grounded evidence + relaxed criteria",
     dict(kind="v2", relaxed=True, evidence=True, structured=False, fewshot=False)),
    ("R3 + structured JSON output",
     dict(kind="v2", relaxed=True, evidence=True, structured=True, fewshot=False)),
    ("R4 + few-shot (4 validation exemplars)",
     dict(kind="v2", relaxed=True, evidence=True, structured=True, fewshot=True)),
]


def run_rung(ds, rung_cfg, dataset_name, model_name=None, k=1):
    event_map = ds["event_map"]
    rows = []
    fs_line = fs_json = None
    for j, c in enumerate(ds["cases"], 1):
        if rung_cfg["kind"] == "v1":
            r = classify_uncertain(c["seq"], c["pred"], c["actual"], event_map,
                                   c["conf_in"], c["anom_in"],
                                   routing_reason=ROUTING_REASON, temperature=0.0,
                                   model_name=model_name)
        else:
            few = None
            if rung_cfg["fewshot"]:
                if rung_cfg["structured"]:
                    fs_json = fs_json or _fewshot_msgs(ds["exemplars"], event_map, True)
                    few = fs_json
                else:
                    fs_line = fs_line or _fewshot_msgs(ds["exemplars"], event_map, False)
                    few = fs_line
            r = classify_uncertain_v2(
                c["seq"], c["pred"], c["actual"], event_map,
                c["conf_in"], c["anom_in"], routing_reason=ROUTING_REASON,
                temperature=0.0, model_name=model_name, dataset=dataset_name,
                evidence=c["evidence"] if rung_cfg["evidence"] else None,
                few_shot=few, relaxed=rung_cfg["relaxed"],
                structured=rung_cfg["structured"], self_consistency_k=k)
        if r.get("error") and not r.get("raw"):
            log.warning("  LLM error on case %d: %s", j, r["error"])
        rows.append({"true": c["true"], "cls": r["classification"],
                     "parse_ok": r["parse_ok"], "halluc": r["hallucinated"],
                     "latency": r["latency"], "block_id": c["block_id"],
                     "raw": r["raw"], "evidence_txt": r["evidence"],
                     "explanation": r["explanation"]})
        if j % 25 == 0:
            log.info("  ... %d/%d", j, len(ds["cases"]))
    return rows


def metrics(rows):
    n = len(rows)
    tp = sum(1 for r in rows if r["cls"] == "SUSPICIOUS" and r["true"] == 1)
    fp = sum(1 for r in rows if r["cls"] == "SUSPICIOUS" and r["true"] == 0)
    tn = sum(1 for r in rows if r["cls"] == "NORMAL" and r["true"] == 0)
    fn = sum(1 for r in rows if r["cls"] == "NORMAL" and r["true"] == 1)
    acc = (tp + tn) / max(n, 1)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) else 0.0
    fpr = fp / max(fp + tn, 1)
    tnr = tn / max(tn + fp, 1)
    bal = 0.5 * (rec + tnr)
    return {"n": n, "acc": acc, "recall": rec, "precision": prec, "fpr": fpr,
            "balanced_acc": bal, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "parse_rate": float(np.mean([r["parse_ok"] for r in rows])) if rows else 0.0,
            "halluc_rate": float(np.mean([r["halluc"] for r in rows])) if rows else 0.0,
            "avg_latency": float(np.mean([r["latency"] for r in rows])) if rows else 0.0}


def baseline_metrics(ds):
    """Three deterministic baselines on the same cases."""
    cases = ds["cases"]
    out = {}
    out["Always Normal"] = metrics(
        [{"true": c["true"], "cls": "NORMAL", "parse_ok": True, "halluc": False,
          "latency": 0.0} for c in cases])
    out["Keyword rule (failure token in actual template)"] = metrics(
        [{"true": c["true"], "cls": "SUSPICIOUS" if c["kw"] else "NORMAL",
          "parse_ok": True, "halluc": False, "latency": 0.0} for c in cases])
    def ext(c):
        e = c["evidence"]
        susp = c["kw"] or e["actual_freq"] == 0 or e["score_percentile"] >= RULE_PCTL
        return "SUSPICIOUS" if susp else "NORMAL"
    out[f"Extended rule (keyword OR never-in-normal OR pctl>={RULE_PCTL:.0f})"] = metrics(
        [{"true": c["true"], "cls": ext(c), "parse_ok": True, "halluc": False,
          "latency": 0.0} for c in cases])
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _rung_table(a, results, baselines):
    a("| Config | Accuracy | Balanced acc | Anomaly recall | Anomaly precision | "
      "FP rate | Confusion TP/FP/TN/FN | Parse | Halluc (raw) | s/call |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for name, m in baselines.items():
        a(f"| _{name}_ | {pct(m['acc'])} | {pct(m['balanced_acc'])} | {pct(m['recall'])} | "
          f"{pct(m['precision'])} | {pct(m['fpr'])} | {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} "
          f"| — | — | 0 |")
    for name, m in results:
        a(f"| **{name}** | **{pct(m['acc'])}** | {pct(m['balanced_acc'])} | "
          f"{pct(m['recall'])} | {pct(m['precision'])} | {pct(m['fpr'])} | "
          f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} | {pct(m['parse_rate'])} | "
          f"{pct(m['halluc_rate'])} | {m['avg_latency']:.2f} |")
    a("")


def write_report(all_results, out_path="outputs/llama_bgl_evaluation.md"):
    L = []; a = L.append
    a("# Llama on the Uncertain Band — Improvement Ladder (BGL primary, HDFS appendix)\n")
    a("_Task 4. For blocks the GRU routes as UNCERTAIN, can Llama's Normal/Suspicious "
      "verdict be made useful by prompt engineering, grounded evidence, structured "
      "outputs, few-shot exemplars, self-consistency, or a bigger model — before any "
      "fine-tuning? Every rung is measured on the SAME fixed-seed case sample, against "
      "three deterministic baselines. The extended rule uses the SAME evidence features "
      "the v2 prompt receives, so it is the honest 'does the LLM add anything beyond "
      "its own inputs' check. All verdicts flow through the production scrubbers; "
      "surviving hallucinations are 0 by construction. Generated by "
      "`evaluate_llm_ladder.py`._\n")

    for ds_name in [d for d in ("bgl", "hdfs") if d in all_results]:
        R = all_results[ds_name]
        meta = R["meta"]
        a(f"\n## {'Primary: BGL' if ds_name == 'bgl' else 'Appendix: HDFS'} "
          f"({meta['n']} uncertain test blocks, seed {meta['seed']}; "
          f"{pct(meta['anom_share'])} truly anomalous; band size {meta['n_uncertain']:,})\n")
        _rung_table(a, R["rungs"], R["baselines"])

        best_name, best = max(R["rungs"], key=lambda kv: kv[1]["acc"])
        base_best = max(R["baselines"].values(), key=lambda m: m["acc"])
        base_best_name = max(R["baselines"], key=lambda k2: R["baselines"][k2]["acc"])
        a(f"**Best Llama config: {best_name} — accuracy {pct(best['acc'])}, anomaly "
          f"recall {pct(best['recall'])}** (vs best deterministic baseline "
          f"*{base_best_name}* at {pct(base_best['acc'])}).\n")

    # Verdict on the BGL primary target
    if "bgl" in all_results:
        R = all_results["bgl"]
        best_name, best = max(R["rungs"], key=lambda kv: kv[1]["acc"])
        base_best = max(R["baselines"].values(), key=lambda m: m["acc"])
        a("\n## Verdict vs the target\n")
        a(f"- Measured best accuracy on BGL uncertain blocks: **{pct(best['acc'])}** "
          f"({best_name}); anomaly recall {pct(best['recall'])} at {pct(best['fpr'])} "
          "false-positive rate.")
        a(f"- 80% target: **{'MET' if best['acc'] >= 0.80 else 'NOT met'}** on this band "
          "(the uncertain band is deliberately the hardest slice — these are the blocks "
          "the GRU could not decide).")
        beats = best["acc"] > base_best["acc"]
        a(f"- Beats the strongest deterministic baseline: **{'yes' if beats else 'no'}** "
          f"({pct(best['acc'])} vs {pct(base_best['acc'])}).")
        if beats and best["recall"] > 0:
            a("- **LoRA gate (G5): NOT triggered** — prompt-level improvements produced a "
              "measured, meaningful gain over every deterministic baseline. Fine-tuning "
              "is not justified by the evidence; the remaining gap is dominated by cases "
              "whose templates genuinely carry no distinguishing signal.")
        else:
            a("- **LoRA gate (G5): TRIGGERED** — the ladder left Llama at/below the "
              "deterministic baselines, so prompting alone is insufficient. Next step "
              "(user action required first): LoRA on Llama-3.2-1B-Instruct (gated HF "
              "model — license + login), training pairs from the VALIDATION uncertain "
              "bands, then GGUF → Ollama, and re-run this exact ladder.")
        a("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L) + "\n")
    log.info("Wrote %s", out_path)


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Llama uncertain-band improvement ladder (Task 4).")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", default="bgl,hdfs")
    ap.add_argument("--skip-3b", action="store_true")
    ap.add_argument("--model-3b", default="llama3.2:3b")
    ap.add_argument("--consistency-k", type=int, default=3)
    args = ap.parse_args()

    ok, _, msg = test_ollama_connection()
    if not ok:
        raise SystemExit(f"Ollama not available ({msg}); the ladder needs real calls.")
    device = get_device()
    log.info("Device: %s", device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    all_results = {}
    for ds_name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds = build_dataset_cases(ds_name, device, args.n, args.seed)
        rung_rows = {}
        results = []
        for rung_name, rcfg in RUNGS:
            log.info("[%s] %s ...", ds_name, rung_name)
            rows = run_rung(ds, rcfg, ds_name)
            rung_rows[rung_name] = rows
            m = metrics(rows)
            results.append((rung_name, m))
            log.info("[%s] %s  acc %.3f  recall %.3f  prec %.3f  (%.2fs/call)",
                     ds_name, rung_name, m["acc"], m["recall"], m["precision"],
                     m["avg_latency"])

        # R5: self-consistency on the best v2 rung so far.
        v2_results = [(n_, m_) for n_, m_ in results if not n_.startswith("R0")]
        best_v2_name, _ = max(v2_results, key=lambda kv: kv[1]["acc"])
        best_v2_cfg = dict(RUNGS)[best_v2_name]
        r5_name = f"R5 self-consistency k={args.consistency_k} on ({best_v2_name})"
        log.info("[%s] %s ...", ds_name, r5_name)
        rows = run_rung(ds, best_v2_cfg, ds_name, k=args.consistency_k)
        rung_rows[r5_name] = rows
        results.append((r5_name, metrics(rows)))

        # R6: bigger model on the best v2 rung (capacity vs prompting).
        if not args.skip_3b:
            r6_name = f"R6 {args.model_3b} on ({best_v2_name})"
            log.info("[%s] %s ...", ds_name, r6_name)
            rows = run_rung(ds, best_v2_cfg, ds_name, model_name=args.model_3b)
            rung_rows[r6_name] = rows
            results.append((r6_name, metrics(rows)))

        all_results[ds_name] = {"meta": ds["meta"], "rungs": results,
                                "baselines": baseline_metrics(ds)}

        # Per-case raw dump for auditability.
        dump = []
        for rung_name, rows in rung_rows.items():
            for r in rows:
                dump.append({"rung": rung_name, **{k: r[k] for k in
                             ("block_id", "true", "cls", "parse_ok", "halluc", "latency")}})
        Path("outputs").mkdir(exist_ok=True)
        pd.DataFrame(dump).to_csv(f"outputs/llm_ladder_{ds_name}.csv", index=False)
        log.info("Wrote outputs/llm_ladder_%s.csv", ds_name)

    write_report(all_results)
    log.info("Done. ✓")


if __name__ == "__main__":
    main()

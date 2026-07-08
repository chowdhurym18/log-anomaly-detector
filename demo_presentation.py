# =============================================================================
# demo_presentation.py — the ONE live demo for the final presentation.
#
# Walks the audience through the FULL production pipeline on ONE real block from
# EACH band, so the whole story is visible end to end:
#
#     Normal      →  auto-cleared (high confidence, low anomaly score)
#     Suspicious  →  flagged + grounded evidence explanation
#     Uncertain   →  escalated to Llama for a final, scrubbed verdict
#
# It reuses the PRODUCTION path exactly (the same functions `main.py --detect`
# uses) — normal-only GRU, block-level surprise `nll_-logp_mean`, and
# validation-derived Normal/Uncertain/Suspicious thresholds. No new model, no new
# scoring rule. Degrades gracefully if Ollama is down (deterministic reports only).
#
# Usage:
#   venv/bin/python demo_presentation.py                       # HDFS
#   venv/bin/python demo_presentation.py --dataset bgl         # BGL
#   venv/bin/python demo_presentation.py --max-blocks 30000    # faster live run
#   venv/bin/python demo_presentation.py --no-llm              # no Ollama needed
#
# Terminology shown (the project's canon):
#   Confidence score = P(actual next event)        (1 − anomaly score)
#   Anomaly score    = surprisal = 1 − P(actual)   (per window, 0–1)
#   Detection score  = nll_-logp_mean               (block-level, sets the bands)
# =============================================================================

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from config import config_for_dataset, get_device, variant_flags
from utils.logging_utils import setup_logging
from utils.format import format_pct as pct, band_label
from preprocessing.encoder import StreamingLabelEncoder
from preprocessing.templates import load_event_templates
from preprocessing import block_dataset as bd
from models.gru_model import GRUAnomalyDetector
from run_scoring_comparison import per_window_signals, build_scores
from pipeline.detection_pipeline import derive_bands, classify, most_surprising_window
from llm.ollama_client import test_ollama_connection
from llm.uncertain_classifier import classify_uncertain
from llm.explanation import generate_explanation

log = logging.getLogger(__name__)

RULE = "═" * 78
THIN = "─" * 78

_BLOCK_WORD = {"hdfs": "one HDFS operation", "bgl": "a 100-line BGL log window"}
_ROUTING_REASON = ("GRU routed this block UNCERTAIN: its surprise score fell between "
                   "the Normal and Suspicious thresholds, so the detector was not "
                   "confident either way.")


def _load(dataset, device):
    """Load the normal-only detector + labelled blocks + templates for `dataset`."""
    cfg = config_for_dataset(dataset)
    encoder = StreamingLabelEncoder.load(Path(cfg["cache_dir"]) / "encoder.pkl")
    event_map = load_event_templates(cfg["datasets"][dataset]["templates"])
    flags = variant_flags("gru")
    model = GRUAnomalyDetector(
        len(encoder.classes_), cfg["embedding_dim"], cfg["hidden_dim"],
        cfg["num_layers"], cfg["dropout"], **flags).to(device)
    model.load_state_dict(torch.load(cfg["detection_checkpoint"], map_location=device))
    model.eval()
    return cfg, encoder, event_map, model


def _pick_examples(bands, test_score, te_y, rng):
    """Pick one representative TEST block per band for the walkthrough.

    Normal    : a correctly-cleared normal block (true label 0), mid-band.
    Suspicious: the most-surprising truly-anomalous block (true label 1) — the
                clearest positive; falls back to any Suspicious block.
    Uncertain : a random Uncertain block (the genuinely ambiguous middle).
    """
    picks = {}

    normal_idx = np.where((bands == "NORMAL") & (te_y == 0))[0]
    if len(normal_idx):
        order = normal_idx[np.argsort(test_score[normal_idx])]
        picks["NORMAL"] = int(order[len(order) // 2])          # median-surprise normal

    susp_idx = np.where(bands == "ANOMALY")[0]
    susp_true = susp_idx[te_y[susp_idx] == 1]
    pool = susp_true if len(susp_true) else susp_idx
    if len(pool):
        picks["ANOMALY"] = int(pool[np.argmax(test_score[pool])])   # most surprising

    unc_idx = np.where(bands == "UNCERTAIN")[0]
    if len(unc_idx):
        picks["UNCERTAIN"] = int(rng.permutation(unc_idx)[0])
    return picks


def _window_facts(model, block, device, cfg, encoder):
    """Decode the block's single most-surprising (context → actual/expected) window."""
    w = most_surprising_window(model, block, device, cfg)
    seq = encoder.inverse_transform(w["context"], context="demo ctx")
    pred = encoder.inverse_transform([w["pred"]], context="demo pred")[0]
    actual = encoder.inverse_transform([w["target"]], context="demo actual")[0]
    return w, seq, pred, actual


def _print_common(stage_title, seq, pred, actual, conf, anom, block_score, susp_thr):
    print(f"  Stage 1 · GRU next-event prediction")
    print(f"    Observed sequence (last {len(seq)} events) : {' '.join(seq)}")
    print(f"    GRU expected next event                    : {pred}")
    print(f"    Actual next event                          : {actual}")
    print(f"    Confidence score  (P of the actual event)  : {pct(conf)}")
    print(f"    Anomaly score     (surprise, 1 − P)        : {pct(anom)}")
    print(f"    Block detection score (nll_-logp_mean)     : {block_score:.3f}"
          f"   (Suspicious threshold {susp_thr:.3f})")


def run(dataset, device, use_llm, max_blocks, seed):
    cfg, encoder, event_map, model = _load(dataset, device)
    rng = np.random.default_rng(seed)

    # ---- Production routing: derive thresholds on VAL, classify TEST ----
    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, cfg, max_blocks=max_blocks)
    _, va_idx, te_idx = bd.split_block_indices(labels, cfg)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, te_ids = bd.select_blocks(blocks, labels, block_ids, te_idx)
    roll = cfg.get("detection_roll_window", 5)
    score_name = cfg.get("detection_score", "nll_-logp_mean")
    val_sig = per_window_signals(model, va_blocks, device, cfg)
    test_sig = per_window_signals(model, te_blocks, device, cfg)
    val_score = build_scores(val_sig, ref_sig=val_sig, roll_window=roll)[score_name]
    test_score = build_scores(test_sig, ref_sig=val_sig, roll_window=roll)[score_name]
    unc_thr, susp_thr = derive_bands(val_score, va_y, cfg.get("detection_max_normal_rate", 0.01))
    bands = classify(test_score, unc_thr, susp_thr)

    n = len(bands)
    n_norm = int((bands == "NORMAL").sum())
    n_unc = int((bands == "UNCERTAIN").sum())
    n_susp = int((bands == "ANOMALY").sum())

    print("\n" + RULE)
    print(f"  LIVE DEMO — Suspicious-Activity Detection on {dataset.upper()}")
    print(f"  (one block = {_BLOCK_WORD.get(dataset, 'one session')}; "
          f"reviewed {n:,} held-out blocks)")
    print(RULE)
    print(f"  The detector sorted every block into three bands, using thresholds it")
    print(f"  learned on a separate validation set (never hand-picked):")
    print(f"    ✅ Normal     {n_norm:>7,}  ({pct(n_norm/n)})  → auto-cleared, no human/LLM")
    print(f"    ❓ Uncertain  {n_unc:>7,}  ({pct(n_unc/n)})  → escalated to Llama")
    print(f"    ⚠  Suspicious {n_susp:>7,}  ({pct(n_susp/n)})  → flagged + explained")
    print(f"\n  Below: one real block from each band, walked end to end.")

    picks = _pick_examples(bands, test_score, np.asarray(te_y), rng)

    # ---------------- NORMAL ----------------
    if "NORMAL" in picks:
        i = picks["NORMAL"]
        w, seq, pred, actual = _window_facts(model, te_blocks[i], device, cfg, encoder)
        print("\n" + THIN)
        print(f"  ✅ NORMAL block  (id {te_ids[i]}, true label: "
              f"{'Anomaly' if te_y[i] else 'Normal'})")
        print(THIN)
        _print_common("", seq, pred, actual, 1.0 - w["surprisal"], w["surprisal"],
                      float(test_score[i]), susp_thr)
        print(f"  Stage 2 · Routing decision")
        print(f"    → {band_label('NORMAL')}: surprise below the auto-clear boundary. "
              f"No LLM call —")
        print(f"      this band is {pct(n_norm/n)} of traffic and costs nothing to clear.")

    # ---------------- SUSPICIOUS ----------------
    if "ANOMALY" in picks:
        i = picks["ANOMALY"]
        w, seq, pred, actual = _window_facts(model, te_blocks[i], device, cfg, encoder)
        print("\n" + THIN)
        print(f"  ⚠  SUSPICIOUS block  (id {te_ids[i]}, true label: "
              f"{'Anomaly' if te_y[i] else 'Normal'})")
        print(THIN)
        _print_common("", seq, pred, actual, 1.0 - w["surprisal"], w["surprisal"],
                      float(test_score[i]), susp_thr)
        print(f"  Stage 2 · Routing decision")
        print(f"    → {band_label('ANOMALY')}: surprise at/above the Suspicious threshold.")
        print(f"  Stage 3 · Grounded evidence explanation (verdict fields are Python-owned;")
        print(f"            the LLM writes ONE scrubbed sentence, or none if Ollama is off):")
        text = generate_explanation(
            sequence=seq, predicted_event=pred, actual_event=actual,
            confidence=1.0 - w["surprisal"], anomaly_score=w["surprisal"],
            classification="ANOMALY", event_context=event_map, use_llm=use_llm)
        for line in text.splitlines():
            print(f"      {line}")

    # ---------------- UNCERTAIN ----------------
    if "UNCERTAIN" in picks:
        i = picks["UNCERTAIN"]
        w, seq, pred, actual = _window_facts(model, te_blocks[i], device, cfg, encoder)
        print("\n" + THIN)
        print(f"  ❓ UNCERTAIN block  (id {te_ids[i]}, true label: "
              f"{'Anomaly' if te_y[i] else 'Normal'})")
        print(THIN)
        _print_common("", seq, pred, actual, 1.0 - w["surprisal"], w["surprisal"],
                      float(test_score[i]), susp_thr)
        print(f"  Stage 2 · Routing decision")
        print(f"    → {band_label('UNCERTAIN')}: the ambiguous middle → escalate to Llama.")
        if use_llm:
            print(f"  Stage 3 · Llama 3.2 FINAL verdict (constrained to NORMAL/SUSPICIOUS,")
            print(f"            grounded only in the templates, then scrubbed):")
            r = classify_uncertain(seq, pred, actual, event_map,
                                   1.0 - w["surprisal"], w["surprisal"],
                                   routing_reason=_ROUTING_REASON, temperature=0.0)
            conf_str = (f" (self-reported confidence {pct(r['confidence'])})"
                        if r["confidence"] is not None else "")
            print(f"      Llama classification : {band_label(r['classification'])}{conf_str}")
            print(f"      Explanation          : {r['explanation']}")
            if r["evidence"]:
                print(f"      Evidence cited       : {r['evidence']}")
        else:
            print(f"  Stage 3 · (Llama skipped: --no-llm). In production this block is")
            print(f"            escalated to the constrained, scrubbed Llama classifier.")

    print("\n" + RULE)
    print(f"  Takeaway: the GRU is the detector; Llama only ever sees the ~{pct((n_unc)/n)} of")
    print(f"  blocks in the Uncertain band, and every explanation is scrubbed of any")
    print(f"  cause not present in the log templates. That is how routing cuts LLM cost")
    print(f"  while keeping the flagged blocks fully explained.")
    print(RULE + "\n")


def main():
    ap = argparse.ArgumentParser(description="Live end-to-end presentation demo (all 3 bands).")
    ap.add_argument("--dataset", default="hdfs", choices=["hdfs", "bgl"])
    ap.add_argument("--no-llm", action="store_true", help="Skip Ollama (deterministic only).")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Subsample blocks for a faster live run (e.g. 30000).")
    ap.add_argument("--seed", type=int, default=0, help="Which Uncertain block to show.")
    args = ap.parse_args()

    setup_logging(level=logging.WARNING)   # quiet — the demo prints its own story
    device = get_device()

    use_llm = not args.no_llm
    if use_llm:
        try:
            ok, model_ok, _ = test_ollama_connection()
            use_llm = bool(ok and model_ok)
        except Exception:
            use_llm = False
        if not use_llm:
            print("  (Ollama not reachable — running deterministic-only; pass --no-llm to silence.)")

    run(args.dataset, device, use_llm, args.max_blocks, args.seed)


if __name__ == "__main__":
    main()

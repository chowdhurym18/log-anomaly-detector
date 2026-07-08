# =============================================================================
# demo.py — Research-fair single-sequence demonstration.
#
# Takes one 20-event window, runs the BiGRU+Attention model, and prints the full
# story for a visitor: predicted vs actual next event, correctness, confidence,
# anomaly score, classification, the ATTENTION WEIGHTS (which events the model
# focused on), and the evidence-based explanation (with the attention section).
#
# Usage:
#   venv/bin/python demo.py                          # random test sequence
#   venv/bin/python demo.py --index 12345            # a specific test sequence
#   venv/bin/python demo.py --events E5 E22 ... E11 --actual E14   # your own window
#   venv/bin/python demo.py --variant bigru_attention --no-llm
#
# Notes:
#   - Uses fixed thresholds (like --quick-test) so no training-set scoring pass is
#     needed — instant for a live demo.
#   - Defaults to the bigru_attention variant (the only one with attention). Train
#     it first:  venv/bin/python main.py --variant bigru_attention
# =============================================================================

import os
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

from config import CONFIG, get_device, variant_flags
from utils.logging_utils import setup_logging
from utils.format import format_pct, band_label, confidence_phrase
from preprocessing.templates import load_event_templates
from models.gru_model import GRUAnomalyDetector
from evaluation.evaluate import composite_score
from evaluation.attention import attention_table, format_attention_table, export_attention_csv
from llm.explanation import generate_explanation
from llm.ollama_client import test_ollama_connection
from main import _load_or_build_cache, checkpoint_dir_for

log = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="Single-sequence research-fair demo.")
    p.add_argument("--variant", default="bigru_attention",
                   choices=["gru", "bigru", "bigru_attention"],
                   help="Model variant to demo (default: bigru_attention).")
    p.add_argument("--index", type=int, default=None,
                   help="Use this exact sequence index from the cached arrays.")
    p.add_argument("--events", nargs="+", default=None, metavar="EID",
                   help="Provide your own window of EventIds (e.g. E5 E22 ... E11).")
    p.add_argument("--actual", default=None, metavar="EID",
                   help="The observed next EventId (required with --events).")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip Ollama; use the deterministic narrative only.")
    return p.parse_args()


def _resolve_sequence(args, encoder, X_path, y_path):
    """Return (window_indices: np.ndarray[int], actual_idx: int|None) for the demo."""
    if args.events:
        if args.actual is None:
            raise SystemExit("--events requires --actual <EventId> (the observed next event).")
        c2i = encoder.class_to_idx
        unknown = [e for e in args.events + [args.actual] if e not in c2i]
        if unknown:
            raise SystemExit(f"Unknown EventId(s) not in vocabulary: {unknown}")
        return np.array([c2i[e] for e in args.events], dtype=np.int64), c2i[args.actual]

    # Otherwise pull a window + its true next event from the cached arrays.
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if args.index is not None:
        idx = args.index
        if not (0 <= idx < X.shape[0]):
            raise SystemExit(f"--index out of range [0, {X.shape[0]}).")
    else:
        idx = int(np.random.default_rng(CONFIG["random_seed"]).integers(0, X.shape[0]))
    return np.asarray(X[idx]).astype(np.int64), int(y[idx])


def main():
    setup_logging(level=logging.WARNING)   # quiet logs — the demo prints its own report
    args   = _parse_args()
    device = get_device()

    event_map = load_event_templates()
    encoder, _, vocab_size, X_path, y_path = _load_or_build_cache(eval_only=True)

    # ---- Load the model variant ----
    flags = variant_flags(args.variant)
    model = GRUAnomalyDetector(
        vocab_size    = vocab_size,
        embedding_dim = CONFIG["embedding_dim"],
        hidden_dim    = CONFIG["hidden_dim"],
        num_layers    = CONFIG["num_layers"],
        dropout       = CONFIG["dropout"],
        bidirectional = flags["bidirectional"],
        use_attention = flags["use_attention"],
    ).to(device)
    best_ckpt = os.path.join(checkpoint_dir_for(args.variant), "best_model.pt")
    if not os.path.exists(best_ckpt):
        raise SystemExit(
            f"No checkpoint at {best_ckpt}. Train it first:\n"
            f"  venv/bin/python main.py --variant {args.variant}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    # ---- Resolve the input window ----
    window_idx, actual_idx = _resolve_sequence(args, encoder, X_path, y_path)
    window_eids = encoder.inverse_transform(window_idx.tolist(), context="demo window")
    actual_eid  = encoder.inverse_transform([actual_idx], context="demo actual")[0]

    # ---- Forward pass (with attention weights) ----
    x = torch.tensor(window_idx, dtype=torch.long, device=device).unsqueeze(0)
    y_true = torch.tensor([actual_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, weights = model(x, return_attention=True)

    pred_idx   = int(logits.argmax(dim=1).item())
    pred_eid   = encoder.inverse_transform([pred_idx], context="demo pred")[0]
    score      = float(composite_score(logits, y_true, CONFIG).item())
    confidence = 1.0 - score
    correct    = (pred_idx == actual_idx)

    # Classification — fixed thresholds (same convention as --quick-test).
    anomaly_threshold   = float(CONFIG["anomaly_threshold"])
    uncertain_threshold = anomaly_threshold * 0.5
    classification = ("ANOMALY"   if score >= anomaly_threshold else
                      "UNCERTAIN" if score >= uncertain_threshold else
                      "NORMAL")

    # ---- Attention table ----
    if weights is not None:
        attn = attention_table(window_eids, weights[0].cpu().numpy())
    else:
        attn = None   # gru/bigru variants have no attention

    # ---- Evidence-based explanation (with the attention-focus section) ----
    use_llm = False
    if not args.no_llm:
        try:
            ok, model_ok, _ = test_ollama_connection()
            use_llm = bool(ok and model_ok)
        except Exception:
            use_llm = False
    explanation = generate_explanation(
        sequence        = window_eids,
        predicted_event = pred_eid,
        actual_event    = actual_eid,
        confidence      = confidence,
        anomaly_score   = score,
        classification  = classification,
        event_context   = event_map,
        use_llm         = use_llm,
        attention       = attn,
    )

    # ---- Print the demo report ----
    print("\n" + "═" * 72)
    print(f"  RESEARCH-FAIR DEMO  —  variant: {args.variant}")
    print("═" * 72)
    print(f"  Input window ({len(window_eids)} events): {' '.join(window_eids)}")
    print("-" * 72)
    print(f"  Predicted next event : {pred_eid}")
    print(f"  Actual next event    : {actual_eid}")
    print(f"  Prediction correct   : {correct}")
    print(f"  Confidence score     : {format_pct(confidence)}  ({confidence_phrase(confidence)})")
    print(f"  Anomaly score        : {format_pct(score)}")
    print(f"  Classification       : {band_label(classification)}")
    if attn is not None:
        print("-" * 72)
        print("  Attention weights (events the model focused on):")
        print(format_attention_table(attn, top_k=len(attn)))
        Path("outputs").mkdir(exist_ok=True)
        export_attention_csv(attn, "outputs/attention_weights.csv")
        print("  (full table exported → outputs/attention_weights.csv)")
    print("-" * 72)
    print("  Evidence-Based Explanation:")
    print()
    print(explanation)
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()

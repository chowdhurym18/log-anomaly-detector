# =============================================================================
# pipeline/uncertain_demo.py — live demo of the two-stage cascade on UNCERTAIN blocks.
#
# For a few blocks the GRU routes as UNCERTAIN, print the FULL chain so it can be
# shown to a professor / audience:
#
#   Log sequence -> GRU (expected vs actual) -> confidence -> anomaly score
#              -> routing decision (UNCERTAIN) -> Llama FINAL classification
#              -> Llama explanation
#
# The GRU stays the primary detector; Llama is the SECOND-STAGE decision maker and
# only ever sees the uncertain band. Reuses the production routing + the constrained,
# scrubbed classifier — no model change, no new logic. Invoked by `main.py
# --demo-uncertain`.
# =============================================================================

import logging

import numpy as np
import torch

from utils.format import format_pct as pct, band_label
from models.gru_model import GRUAnomalyDetector
from config import variant_flags
from preprocessing import block_dataset as bd
from run_scoring_comparison import per_window_signals, build_scores
from pipeline.detection_pipeline import derive_bands, classify, most_surprising_window
from llm.ollama_client import test_ollama_connection
from llm.uncertain_classifier import classify_uncertain

log = logging.getLogger(__name__)

_ROUTING_REASON = ("GRU routed this operation UNCERTAIN: its surprise score fell "
                   "between the Normal and Suspicious thresholds, so the detector "
                   "was not confident either way.")


def run_uncertain_demo(encoder, device, config, event_context=None, n=3, seed=0):
    """Print the end-to-end cascade for `n` random UNCERTAIN blocks."""
    ok, _, msg = test_ollama_connection()
    if not ok:
        log.error("Ollama not available (%s). Start it: `ollama serve` + "
                  "`ollama pull llama3.2:1b`.", msg)
        return

    ckpt = config["detection_checkpoint"]
    flags = variant_flags("gru")
    model = GRUAnomalyDetector(len(encoder.classes_), config["embedding_dim"],
                               config["hidden_dim"], config["num_layers"],
                               config["dropout"], **flags).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log.info("Detector: %s (normal-only GRU) — GRU is the primary detector.", ckpt)

    # Route exactly like production: thresholds on val, bands on test.
    blocks, labels, block_ids = bd.load_labeled_blocks(encoder, config)
    _, va_idx, te_idx = bd.split_block_indices(labels, config)
    va_blocks, va_y, _ = bd.select_blocks(blocks, labels, block_ids, va_idx)
    te_blocks, te_y, te_ids = bd.select_blocks(blocks, labels, block_ids, te_idx)
    roll = config.get("detection_roll_window", 5)
    val_sig = per_window_signals(model, va_blocks, device, config)
    test_sig = per_window_signals(model, te_blocks, device, config)
    val_score = build_scores(val_sig, ref_sig=val_sig, roll_window=roll)["nll_-logp_mean"]
    test_score = build_scores(test_sig, ref_sig=val_sig, roll_window=roll)["nll_-logp_mean"]
    unc, susp = derive_bands(val_score, va_y, config.get("detection_max_normal_rate", 0.01))
    bands = classify(test_score, unc, susp)

    unc_idx = np.where(bands == "UNCERTAIN")[0]
    if len(unc_idx) == 0:
        log.warning("No UNCERTAIN blocks in this split.")
        return
    sample = np.random.default_rng(seed).permutation(unc_idx)[:n]

    print("\n" + "=" * 74)
    print("  TWO-STAGE CASCADE DEMO — GRU routes, Llama adjudicates the UNCERTAIN band")
    print("=" * 74)
    for k, i in enumerate(sample, 1):
        block = te_blocks[i]
        w = most_surprising_window(model, block, device, config)
        seq = encoder.inverse_transform(w["context"], context="demo ctx")
        pred = encoder.inverse_transform([w["pred"]], context="demo pred")[0]
        actual = encoder.inverse_transform([w["target"]], context="demo actual")[0]
        conf, anom = 1.0 - w["surprisal"], w["surprisal"]

        r = classify_uncertain(seq, pred, actual, event_context, conf, anom,
                               routing_reason=_ROUTING_REASON, temperature=0.0)

        print(f"\n── Uncertain block #{k}  (id {te_ids[i]}) " + "─" * 24)
        print(f"  Stage 1 · GRU")
        print(f"    Observed sequence (last {len(seq)} events) : {' '.join(seq)}")
        print(f"    GRU expected next event                    : {pred}")
        print(f"    Actual next event                          : {actual}")
        print(f"    Confidence score  (P of actual)            : {pct(conf)}")
        print(f"    Anomaly score     (surprise)               : {pct(anom)}")
        print(f"    Routing decision                           : {band_label('UNCERTAIN')}"
              f"  → escalate to Llama")
        print(f"  Stage 2 · Llama 3.2 (final decision maker)")
        print(f"    Llama FINAL classification                 : {band_label(r['classification'])}"
              + (f"  (self-reported confidence {pct(r['confidence'])})"
                 if r['confidence'] is not None else ""))
        print(f"    Llama explanation                          : {r['explanation']}")
        if r["evidence"]:
            print(f"    Evidence cited                             : {r['evidence']}")
        if r["insufficient"]:
            print(f"    (Insufficient-evidence fallback used — no guessing.)")
    print("\n" + "=" * 74)
    print("  GRU stayed the primary detector; Llama only saw the uncertain minority,")
    print("  and every explanation is scrubbed of any unsupported cause.")
    print("=" * 74 + "\n")

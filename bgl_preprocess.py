# =============================================================================
# bgl_preprocess.py — turn raw BGL.log into the SAME shape HDFS uses, so the whole
# existing pipeline runs on it unchanged.
#
# BGL has no natural "blocks" (unlike HDFS BlockIds). The standard approach is to
# SESSIONIZE the time-ordered log into fixed-size windows and label a window
# anomalous if it contains >=1 alert line (BGL's first field is "-" for normal,
# else an alert tag). We parse messages into event templates with Drain3 (the same
# class of parser that produced the HDFS templates).
#
# Produces (mirroring data/HDFS_v1/preprocessed/*):
#   data/BGL/BGL_structured.csv   flat one-EventId-per-row (for the encoder)
#   data/BGL/BGL_templates.csv    EventId,EventTemplate (for Llama context)
#   data/BGL/Event_traces.csv     BlockId,Features   (windowed sessions)
#   data/BGL/anomaly_label.csv    BlockId,Label      (Normal/Anomaly)
#   .gru_cache_bgl/encoder.pkl    StreamingLabelEncoder fit on BGL EventIds
#
# Changes nothing in the existing code. Run once:
#   venv/bin/python bgl_preprocess.py [--window 100] [--max-lines N]
# =============================================================================

import os
import csv
import argparse
import logging
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from preprocessing.encoder import StreamingLabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bgl_preprocess")

RAW = "data/BGL/BGL.log"
OUT = "data/BGL"
CACHE = ".gru_cache_bgl"
# BGL columns: Label Timestamp Date Node Time NodeRepeat Type Component Level Content...
N_META = 9   # the first 9 whitespace fields are metadata; the rest is the message


def main():
    ap = argparse.ArgumentParser(description="Preprocess raw BGL.log into HDFS-shaped data.")
    ap.add_argument("--window", type=int, default=100, help="Lines per session/block.")
    ap.add_argument("--max-lines", type=int, default=None, help="Cap lines (default: all).")
    args = ap.parse_args()

    if not Path(RAW).exists():
        raise SystemExit(f"{RAW} not found. Download+extract BGL first.")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(CACHE).mkdir(parents=True, exist_ok=True)

    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    structured_path = os.path.join(OUT, "BGL_structured.csv")
    event_ids, is_anom = [], []           # per-line, in order (for sessionization)

    log.info("Parsing %s with Drain3 ...", RAW)
    with open(RAW, "r", errors="replace") as f, \
         open(structured_path, "w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["EventId", "Label", "Node"])
        for n, line in enumerate(f, 1):
            if args.max_lines and n > args.max_lines:
                break
            parts = line.split()
            if len(parts) <= N_META:
                continue
            label, node = parts[0], parts[3]
            content = " ".join(parts[N_META:])
            res = miner.add_log_message(content)
            eid = f"E{res['cluster_id']}"
            anom = 0 if label == "-" else 1
            event_ids.append(eid); is_anom.append(anom)
            w.writerow([eid, label, node])
            if n % 500000 == 0:
                log.info("  ... %d lines  (templates so far: %d)", n, len(miner.drain.clusters))

    n_lines = len(event_ids)
    log.info("Parsed %d lines into %d templates.", n_lines, len(miner.drain.clusters))

    # --- Templates file (EventId,EventTemplate) ---
    with open(os.path.join(OUT, "BGL_templates.csv"), "w", newline="") as tf:
        w = csv.writer(tf); w.writerow(["EventId", "EventTemplate"])
        for c in miner.drain.clusters:
            w.writerow([f"E{c.cluster_id}", c.get_template()])

    # --- Sessionize into fixed-size windows; label = any-alert-in-window ---
    W = args.window
    n_blocks = n_lines // W
    with open(os.path.join(OUT, "Event_traces.csv"), "w", newline="") as ef, \
         open(os.path.join(OUT, "anomaly_label.csv"), "w", newline="") as lf:
        we = csv.writer(ef); we.writerow(["BlockId", "Features"])
        wl = csv.writer(lf); wl.writerow(["BlockId", "Label"])
        n_anom_blocks = 0
        for b in range(n_blocks):
            s = b * W
            seg = event_ids[s:s + W]
            anom = 1 if any(is_anom[s:s + W]) else 0
            n_anom_blocks += anom
            bid = f"bgl_{b}"
            we.writerow([bid, "[" + ",".join(seg) + "]"])
            wl.writerow([bid, "Anomaly" if anom else "Normal"])

    log.info("Sessions: %d blocks of %d lines; %d anomalous (%.2f%%).",
             n_blocks, W, n_anom_blocks, 100 * n_anom_blocks / max(n_blocks, 1))

    # --- Fit + save a BGL-specific encoder (vocab from BGL EventIds) ---
    enc = StreamingLabelEncoder().fit(structured_path, "EventId", chunksize=200000)
    enc.save(os.path.join(CACHE, "encoder.pkl"))
    log.info("Saved BGL encoder (%d EventIds) → %s/encoder.pkl", len(enc.classes_), CACHE)
    log.info("Done. ✓")


if __name__ == "__main__":
    main()

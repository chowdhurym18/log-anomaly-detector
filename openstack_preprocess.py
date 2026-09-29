# =============================================================================
# openstack_preprocess.py — turn the raw OpenStack logs into the SAME shape HDFS
# and BGL use, so the existing block-level pipeline runs on them unchanged.
#
# Mirrors bgl_preprocess.py in structure and output contract. What differs is
# dictated by the data, and each difference is measured in openstack_audit.py
# (outputs/openstack/dataset_audit.md):
#
#   * FIELD LAYOUT — LogHub's OpenStack format is
#       <Logrecord> <Date> <Time> <Pid> <Level> <Component> [<ADDR>] <Content>
#     (BGL is 9 flat whitespace fields; HDFS arrives pre-parsed.)
#
#   * SESSION KEY — the VM instance UUID, the analogue of HDFS's BlockId. BGL had
#     no natural session key and used fixed 100-line windows; OpenStack does have
#     one, so we use it rather than inventing windows.
#
#   * TWO UUID FORMS — an instance is named either as "[instance: <uuid>]"
#     (compute side) or inside a REST path "/servers/<uuid>" (API side). We accept
#     BOTH ("Strategy B" in the audit): it adds 4,130 events over the compute-only
#     form, raises every labelled anomaly from 25 to 27 events, and adds no new
#     sessions. "req-<uuid>" is a REQUEST id and is never used as a session key.
#
#   * LABELS — only the FOUR instances named in the official anomaly_labels.txt are
#     Anomaly. The other 194 instances in openstack_abnormal.log are NOT documented
#     as anomalous and are labelled Normal. This is an explicit, reversible
#     methodological decision (--label-mode strict|abnormal-file).
#
#   * CRLF — OpenStack_2k.log ships with CRLF endings; every line is rstripped of
#     "\r\n" before parsing so a stray \r can never end up inside a template.
#
# Produces (mirroring data/BGL/*):
#   data/OpenStack/OpenStack_structured.csv  flat one-EventId-per-row (for the encoder)
#   data/OpenStack/OpenStack_templates.csv   EventId,EventTemplate (for LLM context)
#   data/OpenStack/Event_traces.csv          BlockId,Features  (per-instance sessions)
#   data/OpenStack/anomaly_label.csv         BlockId,Label     (Normal/Anomaly)
#   .gru_cache_openstack/encoder.pkl         StreamingLabelEncoder fit on OpenStack
#
# Changes nothing in the existing pipeline code. Run once:
#   venv/bin/python openstack_preprocess.py
# =============================================================================

import os
import re
import csv
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

from preprocessing.encoder import StreamingLabelEncoder
from utils.logging_utils import setup_logging

log = logging.getLogger("openstack_preprocess")

RAW = Path("data/OpenStack")
OUT = Path("data/OpenStack")
CACHE = ".gru_cache_openstack"

# Processed in this order; within an instance, events are finally sorted by
# timestamp so a session that spans two files stays chronological.
SOURCES = ["openstack_normal1.log", "openstack_normal2.log", "openstack_abnormal.log"]

LINE_RE = re.compile(
    r'^(?P<Logrecord>\S+)\s+(?P<Date>\S+)\s+(?P<Time>\S+)\s+(?P<Pid>\S+)\s+'
    r'(?P<Level>\S+)\s+(?P<Component>\S+)\s+\[(?P<ADDR>[^\]]*)\]\s+(?P<Content>.*)$'
)

UUID_PAT = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
INSTANCE_BRACKET = re.compile(r'\[instance:\s*(' + UUID_PAT + r')\]')
INSTANCE_URL = re.compile(r'/servers/(' + UUID_PAT + r')')
UUID_RE = re.compile(UUID_PAT)


def build_miner():
    """Drain3 configured for OpenStack.

    Masking matters here: instance UUIDs, tenant ids, IPs, byte counts and
    durations are all inline parameters. Without masking, Drain would mint a new
    template per VM and the vocabulary would explode from tens to thousands —
    destroying the sequential signal the GRU is supposed to learn. The masks below
    mirror the parameter classes LogPAI's published OpenStack Drain config targets
    (IPs, paths, numbers), plus UUIDs and long hex digests which OpenStack uses
    heavily for instance/image identifiers.
    """
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.5      # LogPAI's published OpenStack Drain similarity threshold
    cfg.drain_depth = 5         # LogPAI's published OpenStack Drain depth
    cfg.masking_instructions = [
        MaskingInstruction(UUID_PAT, "*"),                       # instance/image/request uuids
        MaskingInstruction(r'((\d+\.){3}\d+)', "*"),             # IPv4
        MaskingInstruction(r'\b[0-9a-fA-F]{32,}\b', "*"),        # tenant ids / sha digests
        MaskingInstruction(r'(/[\w.\-]+)+/?', "*"),              # file paths & REST paths
        MaskingInstruction(r'\b\d+\.\d+\b', "*"),                # floats (durations, sizes)
        MaskingInstruction(r'\b\d+\b', "*"),                     # integers
    ]
    return TemplateMiner(config=cfg)


def read_anomaly_ids():
    """The labelled-anomalous instance ids, read verbatim from the official file."""
    path = RAW / "anomaly_labels.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found — the official label file is required.")
    ids = UUID_RE.findall(path.read_text())
    log.info("anomaly_labels.txt: %d labelled anomalous instances", len(ids))
    return set(ids)


def extract_instance(content: str):
    """Return the VM instance uuid for a log line, or None.

    Accepts both syntactic forms (audit Strategy B). Never reads the ADDR field,
    whose "req-<uuid>" is a request id, not an instance."""
    m = INSTANCE_BRACKET.search(content)
    if m:
        return m.group(1)
    m = INSTANCE_URL.search(content)
    if m:
        return m.group(1)
    return None


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Preprocess raw OpenStack logs into HDFS/BGL-shaped data.")
    ap.add_argument("--label-mode", choices=["strict", "abnormal-file"], default="strict",
                    help="strict (default): ONLY the 4 instances in anomaly_labels.txt are "
                         "Anomaly. abnormal-file: every instance appearing in "
                         "openstack_abnormal.log is Anomaly (NOT the documented ground "
                         "truth — provided only so the assumption can be revisited).")
    ap.add_argument("--min-events", type=int, default=2,
                    help="Drop sessions shorter than this (pipeline needs >=1 context->target pair).")
    args = ap.parse_args()

    for s in SOURCES:
        if not (RAW / s).exists():
            raise SystemExit(f"{RAW/s} not found. Extract the official OpenStack.tar.gz into {RAW}/.")
    OUT.mkdir(parents=True, exist_ok=True)
    Path(CACHE).mkdir(parents=True, exist_ok=True)

    anomaly_ids = read_anomaly_ids()
    miner = build_miner()

    structured_path = OUT / "OpenStack_structured.csv"
    # per-instance event list of (timestamp, EventId, source-file)
    sessions = defaultdict(list)
    stats = {"lines": 0, "parsed": 0, "unparsed": 0, "with_instance": 0, "without_instance": 0,
             "per_file": {}}

    log.info("Parsing %d raw files with Drain3 ...", len(SOURCES))
    with open(structured_path, "w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["EventId", "InstanceId", "Timestamp", "Level", "Component", "SourceFile"])
        for src in SOURCES:
            f_lines = f_parsed = f_unparsed = f_inst = 0
            with open(RAW / src, "r", errors="replace") as f:
                for raw in f:
                    f_lines += 1
                    line = raw.rstrip("\r\n")        # CRLF-safe
                    if not line.strip():
                        continue
                    m = LINE_RE.match(line)
                    if not m:
                        # Multi-line Python tracebacks (ERROR/CRITICAL continuation lines)
                        # do not carry the record header. Counted, never silently dropped.
                        f_unparsed += 1
                        continue
                    f_parsed += 1
                    g = m.groupdict()
                    content = g["Content"]
                    inst = extract_instance(content)
                    res = miner.add_log_message(content)
                    eid = f"E{res['cluster_id']}"
                    ts = f'{g["Date"]} {g["Time"]}'
                    w.writerow([eid, inst or "", ts, g["Level"], g["Component"], src])
                    if inst:
                        f_inst += 1
                        sessions[inst].append((ts, eid, src))
            stats["per_file"][src] = {"lines": f_lines, "parsed": f_parsed,
                                      "unparsed": f_unparsed, "with_instance": f_inst}
            stats["lines"] += f_lines; stats["parsed"] += f_parsed
            stats["unparsed"] += f_unparsed; stats["with_instance"] += f_inst
            log.info("  %-28s %7d lines | %6d with instance | %3d unparsed",
                     src, f_lines, f_inst, f_unparsed)

    stats["without_instance"] = stats["parsed"] - stats["with_instance"]
    n_templates = len(miner.drain.clusters)
    log.info("Parsed %d lines into %d templates.", stats["parsed"], n_templates)

    # --- Templates file (EventId,EventTemplate) ---
    with open(OUT / "OpenStack_templates.csv", "w", newline="") as tf:
        w = csv.writer(tf); w.writerow(["EventId", "EventTemplate"])
        for c in miner.drain.clusters:
            w.writerow([f"E{c.cluster_id}", c.get_template()])

    # --- Sessions: sort each instance's events chronologically, then emit ---
    # Sorting matters for the single instance that appears in two files, and is a
    # no-op elsewhere; it guarantees the GRU always sees real temporal order.
    n_short = 0
    rows_traces, rows_labels, rows_meta = [], [], []
    for inst, evs in sessions.items():
        evs.sort(key=lambda t: t[0])
        eids = [e for _, e, _ in evs]
        if len(eids) < args.min_events:
            n_short += 1
            continue
        srcs = sorted({s for _, _, s in evs})
        if args.label_mode == "strict":
            is_anom = inst in anomaly_ids
        else:
            is_anom = "openstack_abnormal.log" in srcs
        rows_traces.append([inst, "[" + ",".join(eids) + "]"])
        rows_labels.append([inst, "Anomaly" if is_anom else "Normal"])
        rows_meta.append([inst, "|".join(srcs), len(eids), evs[0][0], evs[-1][0]])

    with open(OUT / "Event_traces.csv", "w", newline="") as ef:
        w = csv.writer(ef); w.writerow(["BlockId", "Features"]); w.writerows(rows_traces)
    with open(OUT / "anomaly_label.csv", "w", newline="") as lf:
        w = csv.writer(lf); w.writerow(["BlockId", "Label"]); w.writerows(rows_labels)
    # Session provenance — NOT consumed by the detector. Used only by the evaluation
    # phase to break scores down by source file (the provenance/leakage control).
    with open(OUT / "session_meta.csv", "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["BlockId", "SourceFiles", "NEvents", "FirstTs", "LastTs"])
        w.writerows(rows_meta)

    n_anom = sum(1 for r in rows_labels if r[1] == "Anomaly")
    log.info("Sessions: %d (%d anomalous, %.3f%%); dropped %d shorter than %d events.",
             len(rows_traces), n_anom, 100 * n_anom / max(len(rows_traces), 1),
             n_short, args.min_events)

    found = {r[0] for r in rows_labels if r[1] == "Anomaly"}
    if args.label_mode == "strict":
        missing = anomaly_ids - found
        if missing:
            raise SystemExit(f"FATAL: labelled anomalies lost in preprocessing: {missing}")
        log.info("All %d labelled anomalous instances survived preprocessing. ✓", len(anomaly_ids))

    # --- Fit + save an OpenStack-specific encoder ---
    enc = StreamingLabelEncoder().fit(str(structured_path), "EventId", chunksize=200000)
    enc.save(os.path.join(CACHE, "encoder.pkl"))
    log.info("Saved OpenStack encoder (%d EventIds) → %s/encoder.pkl", len(enc.classes_), CACHE)

    stats.update({"templates": n_templates, "sessions": len(rows_traces),
                  "anomalous_sessions": n_anom, "dropped_short_sessions": n_short,
                  "vocab": len(enc.classes_), "label_mode": args.label_mode})
    Path("outputs/openstack").mkdir(parents=True, exist_ok=True)
    with open("outputs/openstack/preprocess_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Wrote outputs/openstack/preprocess_stats.json")
    log.info("Done. ✓")


if __name__ == "__main__":
    main()

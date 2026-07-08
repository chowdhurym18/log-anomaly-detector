"""
hdfs_v1_preprocess.py
=====================
One-time conversion: Event_traces.csv (per-block EventId sequences)
→ HDFS_v1_structured.csv (one EventId per row, same format as HDFS_2k).

Run from the repo root before starting main.py with HDFS_v1:

    python hdfs_v1_preprocess.py

Why this step is needed
-----------------------
The pipeline's StreamingLabelEncoder (preprocessing/encoder.py) and
build_memmap_sequences (preprocessing/dataset.py) both expect a CSV with
one EventId per row (the format used by HDFS_2k).
Event_traces.csv stores each block's full event sequence as a bracketed
list in the "Features" column, e.g. [E5,E22,E5,E11,...].  This script
flattens those per-block sequences into the required row-per-event format.

Block ordering
--------------
Blocks are written in the order they appear in Event_traces.csv.  Events
within each block are written in their original sequence order.
No sentinel is inserted at block boundaries because the existing pipeline
treats the event stream as continuous — identical to how HDFS_2k was
structured as a flat log file without block delimiters.

Output size
-----------
575,061 blocks × avg 19.4 events ≈ 11.2 million rows ≈ 45 MB CSV.
This is well within the streaming pipeline's design budget.
"""

import csv
import re
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_PATH  = "data/HDFS_v1/preprocessed/Event_traces.csv"
OUTPUT_PATH = "data/HDFS_v1/HDFS_v1_structured.csv"

_EVENT_RE   = re.compile(r'E\d+')
_FLUSH_EVERY = 5_000   # blocks between progress reports and write flushes


def convert(input_path: str, output_path: str) -> int:
    """
    Reads Event_traces.csv and writes a flat one-EventId-per-row CSV.

    Uses a write buffer to avoid calling csv.writer.writerow() 11M times
    individually (which is slow due to Python call overhead).  The buffer
    holds up to _FLUSH_EVERY blocks' worth of events, then writes them all
    with a single writerows() call.

    Returns: total number of EventId rows written.
    """
    total_events = 0
    total_blocks = 0
    skipped_rows = 0
    buffer: list[list[str]] = []   # list of single-element rows

    with open(input_path, newline="", encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader, None)
        if header is None:
            log.error("Input file is empty: %s", input_path)
            return 0
        # Validate expected column layout
        if len(header) < 4 or header[3].strip() != "Features":
            log.error(
                "Unexpected CSV layout. Expected column 3 = 'Features', got: %s",
                header
            )
            return 0

        writer.writerow(["EventId"])   # header matching preprocessing/encoder.py

        for row in reader:
            if len(row) < 4:
                skipped_rows += 1
                continue

            events = _EVENT_RE.findall(row[3])
            if not events:
                skipped_rows += 1
                continue

            buffer.extend([ev] for ev in events)
            total_events += len(events)
            total_blocks += 1

            if total_blocks % _FLUSH_EVERY == 0:
                writer.writerows(buffer)
                buffer.clear()
                log.info(
                    "  %s blocks processed  →  %s events written",
                    f"{total_blocks:,}", f"{total_events:,}",
                )

        # Flush remainder
        if buffer:
            writer.writerows(buffer)

    if skipped_rows:
        log.warning("Skipped %s malformed/empty rows.", skipped_rows)

    log.info(
        "Done.  %s blocks  →  %s events  →  %s",
        f"{total_blocks:,}", f"{total_events:,}", output_path
    )
    return total_events


def main() -> None:
    output = Path(OUTPUT_PATH)

    if output.exists():
        log.info(
            "Output already exists at %s  (delete it to regenerate).", OUTPUT_PATH
        )
        sys.exit(0)

    if not Path(INPUT_PATH).exists():
        log.error("Input not found: %s", INPUT_PATH)
        log.error("Expected path: data/HDFS_v1/preprocessed/Event_traces.csv")
        sys.exit(1)

    log.info("Converting %s  →  %s", INPUT_PATH, OUTPUT_PATH)
    n = convert(INPUT_PATH, OUTPUT_PATH)
    if n == 0:
        log.error("No events were written. Check the input file.")
        sys.exit(1)

    log.info(
        "Conversion complete.  %s EventId rows written.", f"{n:,}"
    )
    log.info(
        "Next: delete .gru_cache/ if it exists, then run:  python main.py"
    )


if __name__ == "__main__":
    main()

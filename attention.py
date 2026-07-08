# =============================================================================
# evaluation/attention.py — Turn raw attention weights into a presentable table.
#
# The model's attention layer returns one weight per timestep (per input event).
# For a poster/demo we want a compact "these are the events the model focused on"
# view: weights aggregated per EventId and sorted high→low. Used by demo.py and
# (optionally) the evidence-based explanation report.
# =============================================================================

import csv
from typing import List, Tuple


def attention_table(event_ids: List[str],
                    weights) -> List[Tuple[str, float]]:
    """Aggregate per-timestep attention weights into a per-EventId table.

    The same EventId can appear several times in a window; we SUM its weights so
    the table shows each event's total share of the model's attention. Because
    the raw weights sum to 1 over the window, the aggregated values also sum to 1.

    Args:
        event_ids : decoded EventIds for the window, in order, e.g. ["E5","E22",...].
        weights   : matching attention weights (len == len(event_ids)); a python
                    list or 1-D numpy array.
    Returns:
        [(EventId, weight), ...] sorted by weight descending.
    """
    totals = {}
    for eid, w in zip(event_ids, list(weights)):
        totals[eid] = totals.get(eid, 0.0) + float(w)
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def format_attention_table(table: List[Tuple[str, float]], top_k: int = 5) -> str:
    """Render the top-K rows as aligned 'EventId    Weight' text for the console."""
    rows = ["  EventId    Weight"]
    for eid, w in table[:top_k]:
        rows.append(f"  {eid:<9}  {w:.4f}")
    return "\n".join(rows)


def export_attention_csv(table: List[Tuple[str, float]], path: str) -> None:
    """Write the full (EventId, weight) table to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "attention_weight"])
        for eid, w in table:
            writer.writerow([eid, round(w, 6)])

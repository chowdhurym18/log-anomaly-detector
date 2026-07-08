# =============================================================================
# preprocessing/templates.py — Event template enrichment.
#
# Loads Drain-output EventId → EventTemplate mappings so the LLM receives
# human-readable descriptions instead of opaque symbolic IDs.
# =============================================================================

import os
import re
import logging
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# Candidate files to search, most-specific first.
_TEMPLATE_SEARCH_PATHS = [
    "data/HDFS_v1/preprocessed/HDFS.log_templates.csv",
    "data/HDFS_2k.log_templates.csv",
]


def _clean_template(tmpl: str) -> str:
    """Replace Drain wildcard markers [*] with <...> and strip whitespace."""
    return re.sub(r'\[\*\]', '<...>', tmpl).strip()


def load_event_templates(path: Optional[str] = None) -> Dict[str, str]:
    """
    Loads EventId → EventTemplate mappings from a Drain-output CSV.

    Tries each path in _TEMPLATE_SEARCH_PATHS until it finds a file with
    both an EventId and an EventTemplate column.  Returns an empty dict
    (graceful degradation) when nothing is found, so callers that pass
    the result as event_context fall back to the existing ID-only prompts.

    Args:
        path: Explicit CSV path override. Auto-detects when None.

    Returns:
        Dict mapping e.g. "E11" → "PacketResponder <...> for block <...> terminating"
    """
    candidates = [path] if path else _TEMPLATE_SEARCH_PATHS

    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            df = pd.read_csv(candidate)
            if "EventId" not in df.columns or "EventTemplate" not in df.columns:
                log.debug(
                    "Skipping %s — missing EventId or EventTemplate column", candidate
                )
                continue
            event_map: Dict[str, str] = {
                str(row["EventId"]): _clean_template(str(row["EventTemplate"]))
                for _, row in df.iterrows()
            }
            log.info("Loaded %d event templates from %s", len(event_map), candidate)
            return event_map
        except Exception as exc:
            log.warning("Failed to load templates from %s: %s", candidate, exc)

    log.warning(
        "No event template file found (searched: %s). "
        "LLM explanations will use raw EventIds.",
        ", ".join(c for c in candidates if c),
    )
    return {}


def decode_sequence(
    sequence: List[str],
    event_map: Dict[str, str],
) -> List[str]:
    """
    Replaces EventIds with their human-readable log templates.

    Falls back to the original EventId string for any ID not in event_map,
    so the function is safe to call even with a partial or empty mapping.

    Args:
        sequence:  List of EventId strings, e.g. ["E23", "E21", "E9"].
        event_map: Mapping returned by load_event_templates().

    Returns:
        List of template strings (or original EventId when template absent).

    Example:
        >>> m = {"E23": "Block invalidated on <...>", "E21": "Deleting block <...>"}
        >>> decode_sequence(["E23", "E21"], m)
        ["Block invalidated on <...>", "Deleting block <...>"]
    """
    return [event_map.get(eid, eid) for eid in sequence]


def test_event_template_mapping(event_map: Dict[str, str]) -> None:
    """
    Logs every EventId → template pair so the mapping can be visually verified.
    Call this once after load_event_templates() to confirm templates loaded.

    (Despite the name, this is a startup diagnostic, not a unit test.)
    """
    if not event_map:
        log.warning("test_event_template_mapping: event_map is empty — no templates loaded.")
        return
    log.info("Event template mapping (%d entries):", len(event_map))
    for eid in sorted(event_map, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        log.info("  EventId: %-5s  Template: %s", eid, event_map[eid])

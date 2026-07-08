# =============================================================================
# utils/format.py — tiny shared helpers for HUMAN-READABLE output (Task 5).
#
# The model works in [0,1] scores; people read percentages and words. These
# helpers turn a raw score like 0.4946 into "49.5%" and an internal band enum
# into the presentation label ("Suspicious"), so console output and reports are
# understandable by a non-technical audience (professors, students, fair visitors)
# without changing any stored CSV schema or internal logic.
# =============================================================================


def format_pct(x, digits: int = 1) -> str:
    """0.4946 -> '49.5%'. For showing a [0,1] score to a human."""
    return f"{float(x) * 100:.{digits}f}%"


def confidence_phrase(confidence: float) -> str:
    """One-word reading of a [0,1] confidence (how sure the model is the activity
    is normal): high / moderate / low. Mirrors the routing bands."""
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.40:
        return "moderate"
    return "low"


# Internal enum → human-facing band name. The codebase keeps ANOMALY/UNCERTAIN/
# NORMAL internally (CSV schema, tests); humans see "Suspicious"/"Uncertain"/"Normal".
_BAND_DISPLAY = {"ANOMALY": "Suspicious", "UNCERTAIN": "Uncertain", "NORMAL": "Normal"}


def band_label(internal: str) -> str:
    """'ANOMALY' -> 'Suspicious' (presentation layer; internal enum is unchanged)."""
    return _BAND_DISPLAY.get(str(internal).upper().strip(), str(internal))

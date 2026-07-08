# ===========================================================================
# llm/explanation.py — EVIDENCE-BASED EXPLANATION LAYER
# ---------------------------------------------------------------------------
# The deterministic verdict fields (Expected/Observed event, Prediction
# Correct, Confidence, Anomaly Score, Classification, Root Cause Assessment,
# Recommended Investigation) are assembled in PYTHON from the facts + calibrated
# detector thresholds — never by the model. The LLM is confined to ONE narrative
# field (Evidence-Based Interpretation), which is then scrubbed of verdict
# restatements, speculation language, and any cause it cannot support from the
# event templates. This makes self-contradiction and root-cause hallucination
# structurally impossible.
# ===========================================================================

import re
import time
import logging
from typing import Dict, List, Optional

from config import CONFIG
from llm.ollama_client import (
    chat,
    _llm_stats,
    _llm_stats_lock,
    _explanation_cache,
    _make_cache_key,
    _apply_rate_limit,
    test_ollama_connection,
    log_llm_stats,
)

log = logging.getLogger(__name__)

# Canonical "no root cause" sentence (required verbatim by the spec).
_INSUFFICIENT_EVIDENCE = (
    "The sequence does not provide enough evidence to determine a root cause."
)

# Root causes the model must NEVER introduce unless the literal term appears in
# an involved event template. Single words matched as whole words (case-insensitive).
_BANNED_CAUSE_TERMS = (
    "disk", "hardware", "memory", "ram", "cpu", "swap", "oom",
    "network", "congestion", "bandwidth", "latency", "packet", "firewall",
    "dns", "connectivity", "corrupt", "corruption", "config", "configuration",
    "misconfiguration", "misconfigured", "permission", "credential",
    "security", "attack", "malicious", "intrusion", "overload", "outage",
    "power", "temperature", "throttle", "capacity", "quota",
    "throughput", "bottleneck", "storage", "infrastructure",
    "failure", "software", "resource", "performance",
)

# Multi-word banned phrases — substring-matched against lowercased sentence text.
_BANNED_CAUSE_PHRASES = (
    "software bug", "resource constraint", "resource issue",
    "performance issue", "configuration error", "data management",
    "service outage", "system failure",
)

# Tokens checked against the actual event's template for Root Cause Assessment.
# Exact set from requirement 6 — matched as whole words.
_FAILURE_TEMPLATE_TOKENS = (
    "exception", "error", "failed", "failure", "timeout", "denied",
)

# System prompt: single narrative section, evidence-strict, no verdicts, no speculation.
# It asks for a CONCRETE, achievable task — a factual comparison of the expected vs
# actual templates — so the 1B model produces text that survives the scrubbers,
# instead of vague causal guesses that get deleted.
_EVIDENCE_SYSTEM_PROMPT = """\
You are an evidence-based log analysis assistant for HDFS (Hadoop Distributed
File System). You receive verified facts from a GRU next-event detector: an
ordered event sequence, each event's Drain log template, the expected next event,
the actual next event, and the detector's confidence and anomaly score. Everything
you write must come from those facts and nothing else.

YOUR TASK: write ONE short, factual paragraph (2-4 sentences) that COMPARES the
expected event's template with the actual event's template. Say what operation
each template records, and state whether the actual event is a consistent
continuation of the events already shown in the sequence. Describe only what the
templates literally say.

You MUST NOT:
- invent or imply a root cause of any kind;
- claim malware, an intrusion, or any attack;
- claim a network problem;
- claim a hardware failure;
- mention disk, memory, CPU, network, bandwidth, corruption, configuration,
  permissions, security, capacity, storage, software, resources, or performance
  UNLESS that exact word already appears in one of the templates shown to you;
- make any assumption about system state that is not present in the evidence;
- use speculation words: "suggests", "indicates", "implies", "may", "could",
  "likely", "possibly", "potentially", "appears to", "seems to";
- restate the confidence score, anomaly score, classification, or whether the
  prediction was correct — those are reported separately.

If the templates do not let you say anything factual about the difference, write
exactly: "The templates do not describe a difference that explains this deviation."

Output ONLY this one section, nothing else:

EVIDENCE_BASED_INTERPRETATION:
<your factual template comparison>\
"""

# User prompt: facts in (as an explicit Step 1-4 workflow), single narrative
# section out (Step 5). The EVIDENCE_BASED_INTERPRETATION: header is unchanged so
# the existing parser/scrubbers keep working verbatim.
_EVIDENCE_USER_TEMPLATE = """\
BEGIN_FACTS
Observed sequence ({n_events} events, in order):
{sequence_with_templates}

Expected next event : {pred_event} -> {pred_template}
Actual next event   : {actual_event} -> {actual_template}
Detector context (do NOT restate these numbers): confidence={confidence:.2f}, anomaly_score={anomaly_score:.2f}
END_FACTS

Everything outside BEGIN_FACTS / END_FACTS is unknown. Compare the expected and
actual templates above and describe the difference factually (what each step
records, and whether the actual event fits the sequence so far).

EVIDENCE_BASED_INTERPRETATION:
...\
"""


# ---------------------------------------------------------------------------
# Evidence helpers — pure, deterministic, no model involvement.
# ---------------------------------------------------------------------------
def _norm_id(eid) -> str:
    """Normalise an EventId to a comparable string (strip/whitespace)."""
    return str(eid).strip()


def _involved_templates(sequence: List[str], predicted_event: str,
                        actual_event: str,
                        event_context: Optional[Dict[str, str]]) -> Dict[str, str]:
    """{EventId: template} for every event in the case (sequence + pred + actual)."""
    ctx = event_context or {}
    ids = list(dict.fromkeys(
        [_norm_id(e) for e in sequence]
        + [_norm_id(predicted_event), _norm_id(actual_event)]
    ))
    return {eid: ctx.get(eid, "") for eid in ids}


def _templates_blob(involved: Dict[str, str]) -> str:
    """Lower-cased concatenation of all involved template text (evidence corpus)."""
    return " ".join(t for t in involved.values() if t).lower()


def _term_in_templates(term: str, blob: str) -> bool:
    """True iff `term` appears as a whole word anywhere in the involved templates."""
    return re.search(rf"\b{re.escape(term)}\b", blob) is not None


# Verdict keywords the LLM must never restate; any sentence containing one is dropped.
_VERDICT_PHRASES = (
    "prediction correct", "confidence score", "anomaly score", "classification",
    "expected event", "observed event", "workflow transition",
    "expected next event", "observed next event",
)

# Speculation language banned from LLM narrative (no causal inference allowed).
_SPECULATION_PHRASES = (
    "suggests", "indicates", "implies", "may indicate", "could indicate",
    "likely", "possibly", "potentially", "appears to", "seems to",
)


def _scrub_verdict_phrases(text: str) -> str:
    """Drop any sentence that echoes a structured verdict keyword."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences
            if s.strip() and not any(p in s.lower() for p in _VERDICT_PHRASES)]
    return " ".join(kept).strip()


def _scrub_speculation(text: str) -> str:
    """Drop any sentence containing speculation / causal-inference language."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences
            if s.strip() and not any(p in s.lower() for p in _SPECULATION_PHRASES)]
    return " ".join(kept).strip()


# Phrases that imply a workflow deviation. When the prediction is CORRECT
# (expected == actual) NONE of these can be true, so any sentence containing one
# is a contradiction of the deterministic facts and is dropped (Task 5). Scoped to
# the LLM narrative only — never applied to the deterministic Deviation Analysis,
# which legitimately contains the word "deviation" in the required sentence.
_CONTRADICTION_PHRASES = (
    "deviation", "deviates", "deviated", "different event", "differs from",
    "instead of", "unexpected event", "unexpected transition", "does not match",
    "doesn't match", "did not match", "didn't match", "mismatch",
    "anomalous transition", "out of order", "out-of-order", "wrong event",
)


def _scrub_contradiction(text: str, prediction_correct: bool) -> str:
    """When expected == actual, drop any sentence implying a workflow deviation.

    Guarantees the LLM narrative can never contradict the deterministic fact that
    no event-order deviation occurred. A no-op when prediction_correct is False
    (a real deviation case, where such language is accurate)."""
    if not text or not prediction_correct:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences
            if s.strip() and not any(p in s.lower() for p in _CONTRADICTION_PHRASES)]
    return " ".join(kept).strip()


def _scrub_narrative(text: str, blob: str) -> str:
    """Pipeline scrubber: verdict phrases → speculation → banned cause terms.

    Guarantees the model narrative cannot smuggle in verdict restatements,
    speculation language, or root causes (disk / memory / network / etc.) that
    are not literally present in the involved event templates."""
    text = _scrub_verdict_phrases(text)
    text = _scrub_speculation(text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for s in sentences:
        low = s.lower()
        unsupported_terms = [t for t in _BANNED_CAUSE_TERMS
                             if re.search(rf"\b{re.escape(t)}\b", low)
                             and not _term_in_templates(t, blob)]
        unsupported_phrases = [p for p in _BANNED_CAUSE_PHRASES
                               if p in low and p not in blob]
        if unsupported_terms or unsupported_phrases:
            continue
        if s.strip():
            kept.append(s.strip())
    return " ".join(kept).strip()


def _parse_narrative_sections(raw: str) -> str:
    """Extract EVIDENCE_BASED_INTERPRETATION from the model output.

    Any other sections (including OBSERVED_SEQUENCE_PATTERN if the model
    produces one anyway) are discarded. Returns '' if the section is absent."""
    if not raw:
        return ""
    m = re.search(r"EVIDENCE_BASED_INTERPRETATION\s*:?\s*(.*)$",
                  raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _default_observed_pattern(sequence: List[str],
                              involved: Dict[str, str]) -> str:
    """Deterministic, template-grounded description of the sequence."""
    ids = [_norm_id(e) for e in sequence]
    if not ids:
        return "The observed window contains no events."
    if any(involved.get(e) for e in ids):
        shown = ids[-5:]
        parts = [f"{e} ({involved.get(e) or 'template unavailable'})" for e in shown]
        lead = "" if len(ids) <= 5 else f"{len(ids)} events ending in "
        return (f"The sequence is {lead}" + " -> ".join(parts) + ".").replace("is  ", "is ")
    return (f"The sequence is {len(ids)} events: " + " -> ".join(ids)
            + ". No log templates were supplied for these EventIds.")


def _default_interpretation(predicted_event: str, actual_event: str,
                            prediction_correct: bool,
                            involved: Dict[str, str]) -> str:
    """Deterministic interpretation grounded only in expected vs. observed.

    Used verbatim when the LLM is unavailable or its narrative is fully scrubbed.
    Kept factual and template-grounded (no causes) so the fallback is still useful."""
    p, a = _norm_id(predicted_event), _norm_id(actual_event)
    pt, at = involved.get(p, ""), involved.get(a, "")
    if prediction_correct:
        return (f"The observed event {a}"
                + (f" (\"{at}\")" if at else "")
                + " is the model's own top prediction, so the logged step matches "
                "expectation. This case was flagged because the model's certainty was "
                "low, even though the predicted and actual events are the same.")
    base = (f"The expected step was {p}"
            + (f" (\"{pt}\")" if pt else "")
            + f", but the sequence recorded {a}"
            + (f" (\"{at}\")" if at else "")
            + f" instead — a deviation at the {p} -> {a} transition.")
    if not (pt and at):
        base += " " + _INSUFFICIENT_EVIDENCE
    return base


def _deviation_analysis(predicted_event: str, actual_event: str,
                        prediction_correct: bool,
                        involved: Dict[str, str]) -> str:
    """Deterministic, template-grounded description of the workflow transition.

    Describes WHAT changed (expected step vs. actual step) — never WHY. It quotes
    only the event templates supplied for this case, so it cannot invent a cause.
    This is one of the new sections that make the report informative even when the
    LLM narrative is empty."""
    p, a = _norm_id(predicted_event), _norm_id(actual_event)
    pt, at = involved.get(p, ""), involved.get(a, "")
    if prediction_correct:
        msg = (f"No workflow deviation was detected. The observed next event {a} matches "
               f"the model's top prediction, so the event order did not change. This case "
               f"was flagged by the detector's low certainty, not by any change in the "
               f"sequence of events")
        return msg + (f". Recorded operation: \"{at}\"." if at else ".")
    base = (f"The model expected {p}"
            + (f" (\"{pt}\")" if pt else "")
            + f", but the sequence continued with {a}"
            + (f" (\"{at}\")" if at else "")
            + f". In workflow terms this is the {p} -> {a} transition.")
    if pt and at:
        base += (" The templates show what each step records; they do not by themselves "
                 "state why the order changed.")
    else:
        base += " " + _INSUFFICIENT_EVIDENCE
    return base


def _confidence_analysis(confidence: float, anomaly_score: float,
                         classification: str) -> str:
    """Plain-language meaning of the detector's confidence, written for a
    non-technical reader. Reports confidence as a percentage and says, in one
    sentence, what the detector did about it. Grounded purely in the numbers — no
    causes, no speculation, no raw decimals (Task 6)."""
    pct = f"{confidence * 100:.1f}%"
    if classification == "ANOMALY":
        return (f"The model had very low confidence in this prediction ({pct}), so the "
                "case was automatically flagged as a likely anomaly for review.")
    if classification == "NORMAL":
        return (f"The model was confident in this prediction ({pct}), so the case looks "
                "like normal activity.")
    return (f"The model is uncertain about this prediction ({pct}) and therefore routed "
            "this case for additional review.")


def _root_cause_assessment(actual_event: str, involved: Dict[str, str]) -> str:
    """Exact two-case rule (requirement 6):
    - If the actual event's template contains a failure token → name it.
    - Otherwise → standard insufficient-evidence sentence.
    Only the actual event's template is checked; other templates are ignored."""
    actual_tmpl = involved.get(_norm_id(actual_event), "").lower()
    hit = next((tok for tok in _FAILURE_TEMPLATE_TOKENS
                if re.search(rf"\b{re.escape(tok)}\b", actual_tmpl)), None)
    if hit:
        return (f"The observed event template contains \"{hit}\". "
                "The sequence does not provide enough evidence to determine "
                "a deeper root cause.")
    return _INSUFFICIENT_EVIDENCE


def _recommended_investigation(predicted_event: str, actual_event: str,
                               prediction_correct: bool,
                               involved: Dict[str, str]) -> List[str]:
    """Investigation steps grounded strictly in the case's own events/templates."""
    p, a = _norm_id(predicted_event), _norm_id(actual_event)
    pt, at = involved.get(p, ""), involved.get(a, "")
    steps = [f"Review the raw HDFS log lines matching the observed event {a}"
             + (f": \"{at}\"." if at else ".")]
    if not prediction_correct:
        steps.append(f"Compare against the expected event {p}"
                     + (f": \"{pt}\"." if pt else ".")
                     + f" Confirm whether the {p} -> {a} transition is valid for this block.")
    else:
        steps.append(f"Confirm why the anomaly score is elevated even though the "
                     f"predicted and actual events both equal {a}.")
    steps.append("Verify whether this exact event sequence has occurred in "
                 "known-normal traffic for the same BlockId.")
    return steps


def _assemble_evidence_report(*, sequence: List[str], predicted_event: str,
                              actual_event: str, confidence: float,
                              anomaly_score: float, classification: str,
                              prediction_correct: bool,
                              involved: Dict[str, str],
                              interpretation: str,
                              attention=None) -> str:
    """Render the final report. All verdict fields are Python-deterministic.
    `interpretation` is the only LLM-supplied field (already scrubbed).
    `observed_pattern`, `root_cause_assessment`, and the optional attention
    section are always Python-generated.

    `attention`, when supplied, is a list of (EventId, weight) tuples (already
    aggregated + sorted) from the attention model; it renders a deterministic
    "Model Attention Focus" section. When None (the default and every current
    caller), the section is omitted and the report is byte-identical to before."""
    p, a = _norm_id(predicted_event), _norm_id(actual_event)
    pt = involved.get(p, "") or "template unavailable"
    at = involved.get(a, "") or "template unavailable"
    # Final guarantee: if the prediction is correct, no narrative sentence may imply
    # a deviation. Falls back to a safe deterministic line if everything is scrubbed.
    interpretation = _scrub_contradiction(interpretation, prediction_correct)
    if not interpretation:
        interpretation = ("The predicted and actual events match; no workflow deviation "
                          "was detected. The case was flagged by low model certainty.")
    observed_pattern = _default_observed_pattern(sequence, involved)
    deviation = _deviation_analysis(predicted_event, actual_event,
                                    prediction_correct, involved)
    conf_analysis = _confidence_analysis(confidence, anomaly_score, classification)
    rc = _root_cause_assessment(actual_event, involved)
    steps = _recommended_investigation(predicted_event, actual_event,
                                       prediction_correct, involved)
    inv = "\n".join(f"  * {s}" for s in steps)

    # Optional, Python-owned attention section (the LLM never touches it).
    attention_block = ""
    if attention:
        rows = "\n".join(f"  {_norm_id(eid):<9}  {float(w):.4f}" for eid, w in attention[:5])
        attention_block = (
            "Model Attention Focus\n"
            "  (input events the model weighted most when predicting the next event)\n"
            f"{rows}\n\n"
        )

    return (
        "Summary\n"
        f"  Observed Sequence Pattern: {observed_pattern}\n\n"
        "Prediction\n"
        f"  Expected Event    : {p} — {pt}\n"
        f"  Observed Event    : {a} — {at}\n"
        f"  Prediction Correct: {prediction_correct}\n\n"
        "Deviation Analysis\n"
        f"  {deviation}\n\n"
        "Confidence\n"
        f"  Confidence Score  : {confidence * 100:.1f}%\n"
        f"  Anomaly Score     : {anomaly_score * 100:.1f}%\n"
        f"  Classification    : {classification}\n"
        f"  Interpretation    : {conf_analysis}\n\n"
        "Evidence-Based Interpretation\n"
        f"  {interpretation}\n\n"
        f"{attention_block}"
        "Root Cause Assessment\n"
        f"  {rc}\n\n"
        "Recommended Investigation\n"
        f"{inv}"
    )


def generate_explanation(
    sequence: List[str],
    predicted_event: str,
    actual_event: str,
    confidence: float,
    anomaly_score: float,
    classification: str,
    event_context: Optional[Dict[str, str]] = None,
    use_llm: bool = True,
    attention=None,
) -> str:
    """
    Strictly evidence-based explanation for a flagged sequence.

    Design (anti-hallucination, anti-contradiction):
      * Every verdict field — Prediction Correct, Expected/Observed event,
        Confidence Score, Anomaly Score, Classification, Root Cause Assessment,
        Recommended Investigation — is computed deterministically in Python from
        the supplied facts + the calibrated detector classification. The model
        cannot alter or contradict them.
      * `prediction_correct` is recomputed here as (predicted == actual); the
        model is never asked to judge it.
      * `classification` must already be the calibrated label from
        choose_thresholds() (ANOMALY / UNCERTAIN / NORMAL); it is sanitised and
        echoed verbatim, never re-derived from a hardcoded 0.5 cutoff.
      * The LLM only writes one narrative field (Evidence-Based Interpretation),
        which is scrubbed of verdict restatements, speculation language, and any
        banned root cause not literally present in the involved event templates.
      * If Ollama is unavailable, a deterministic default fills the narrative
        field, so a valid report is always produced.

    Args:
        sequence:        Decoded EventId window.
        predicted_event: EventId the GRU expected next.
        actual_event:    EventId that actually occurred.
        confidence:      Detector confidence (0–1).
        anomaly_score:   Detector anomaly score (0–1).
        classification:  Calibrated label: "ANOMALY" | "UNCERTAIN" | "NORMAL".
        event_context:   EventId → template mapping (load_event_templates()).

    Returns:
        Report string in the required evidence-based format.
    """
    # --- Sanitise / validate the deterministic inputs ---
    classification = str(classification).upper().strip()
    if classification not in ("ANOMALY", "UNCERTAIN", "NORMAL"):
        log.warning("Unexpected classification %r; defaulting to UNCERTAIN. It "
                    "must come from choose_thresholds().", classification)
        classification = "UNCERTAIN"

    # Authoritative correctness check (requirement 6) — never trust the model.
    prediction_correct = (_norm_id(predicted_event) == _norm_id(actual_event))

    involved = _involved_templates(sequence, predicted_event, actual_event, event_context)
    blob     = _templates_blob(involved)

    # --- Cache lookup ---
    cache_key = _make_cache_key(sequence, predicted_event, actual_event,
                                confidence, anomaly_score, classification)
    if cache_key in _explanation_cache:
        with _llm_stats_lock:
            _llm_stats["total_cached"] += 1
        log.debug("Explanation cache hit for key %s", cache_key[:8])
        return _explanation_cache[cache_key]

    # --- Deterministic interpretation default (used as-is if the LLM is absent) ---
    interpretation = _default_interpretation(predicted_event, actual_event,
                                             prediction_correct, involved)

    # --- Optional LLM narrative (EVIDENCE_BASED_INTERPRETATION only, scrubbed) ---
    # When the prediction is CORRECT (expected == actual) there is no template
    # difference to compare, and the deterministic sections already state that no
    # workflow deviation occurred. Skipping the LLM here makes it structurally
    # impossible for the model to contradict that fact, and saves a call (Task 5).
    if chat is not None and use_llm and not prediction_correct:
        seq_lines = [
            f"  {idx:2d}. {_norm_id(eid)} -> {involved.get(_norm_id(eid)) or 'template unavailable'}"
            for idx, eid in enumerate(sequence, 1)
        ]
        user_prompt = _EVIDENCE_USER_TEMPLATE.format(
            n_events                = len(sequence),
            sequence_with_templates = "\n".join(seq_lines),
            pred_event              = _norm_id(predicted_event),
            pred_template           = involved.get(_norm_id(predicted_event)) or "template unavailable",
            actual_event            = _norm_id(actual_event),
            actual_template         = involved.get(_norm_id(actual_event)) or "template unavailable",
            confidence              = confidence,
            anomaly_score           = anomaly_score,
        )

        model_name   = CONFIG.get("ollama_model", "llama3.2:1b")
        retries      = CONFIG.get("ollama_retries", 3)
        backoff      = CONFIG.get("ollama_backoff", 1.5)
        min_interval = CONFIG.get("ollama_rate_limit_seconds", 0.5)
        _apply_rate_limit(min_interval)

        for attempt in range(1, retries + 1):
            # Time each chat() attempt so we can report average LLM latency.
            call_t0 = time.perf_counter()
            try:
                with _llm_stats_lock:
                    _llm_stats["total_llm_calls"] += 1
                resp = chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": _EVIDENCE_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                with _llm_stats_lock:
                    _llm_stats["total_latency_seconds"] += time.perf_counter() - call_t0
                try:
                    raw = resp["message"]["content"].strip()
                except Exception:
                    raw = str(resp).strip()

                int_raw = _parse_narrative_sections(raw)
                int_clean = _scrub_narrative(int_raw, blob)
                if int_clean:
                    interpretation = int_clean
                break
            except Exception as exc:
                # Record the time spent on the failed attempt and count the failure.
                with _llm_stats_lock:
                    _llm_stats["total_latency_seconds"] += time.perf_counter() - call_t0
                    _llm_stats["total_failures"] += 1
                log.warning("Ollama chat attempt %s/%s failed: %s",
                            attempt, retries, repr(exc))
                if attempt < retries:
                    time.sleep(backoff ** attempt)
                # else: fall through to deterministic default already set.

    report = _assemble_evidence_report(
        sequence           = sequence,
        predicted_event    = predicted_event,
        actual_event       = actual_event,
        confidence         = confidence,
        anomaly_score      = anomaly_score,
        classification     = classification,
        prediction_correct = prediction_correct,
        involved           = involved,
        interpretation     = interpretation,
        attention          = attention,
    )
    _explanation_cache[cache_key] = report
    return report


def send_to_llm(anomaly_cases: List[dict],
                event_context: Optional[Dict[str, str]] = None) -> None:
    """
    Called for HIGH-CONFIDENCE anomalies.  Generates a grounded explanation for
    each case via Ollama.  Repeated anomaly patterns are served from cache.

    Args:
        anomaly_cases:  Decoded anomaly records with model metadata.
        event_context:  Optional EventId → description mapping.  Pass it here
                        and it will be forwarded to generate_explanation() so
                        the LLM can ground its response in real descriptions.
                        Example:
                            {
                                "E6":  "HDFS block report received",
                                "E11": "Namenode received block allocations",
                            }
    """
    if chat is None:
        log.warning("Ollama Python client not installed; skipping LLM explanations.")
        return

    try:
        ok, model_ok, msg = test_ollama_connection()
    except Exception as exc:
        log.warning("Ollama connectivity check failed: %s", repr(exc))
        ok = False
        msg = repr(exc)

    # Even when Ollama is down we still emit deterministic, evidence-based
    # reports (the narrative falls back to template-grounded defaults) rather
    # than skipping — the verdict fields never needed the model.
    llm_ok = bool(ok and model_ok)
    if not llm_ok:
        log.warning(
            "Ollama unavailable or model missing (%s); emitting deterministic "
            "evidence-based reports without the LLM narrative.", msg
        )

    with _llm_stats_lock:
        _llm_stats["total_anomalies"] += len(anomaly_cases)

    for case in anomaly_cases:
        sequence = case["sequence"]
        explanation = generate_explanation(
            sequence        = sequence,
            predicted_event = case["predicted_event"],
            actual_event    = case["actual_event"],
            confidence      = case["confidence"],
            anomaly_score   = case["anomaly_score"],
            classification  = case["classification"],
            event_context   = event_context,
            use_llm         = llm_ok,
        )
        tag = "[→ LLM]" if llm_ok else "[→ LLM - DETERMINISTIC]"
        log.info("%s  Sequence: %s", tag, sequence)
        log.info("Explanation:\n%s", explanation)

    log_llm_stats()

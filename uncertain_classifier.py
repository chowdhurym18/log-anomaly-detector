# =============================================================================

# =============================================================================

import re
import json
import time
import logging

from config import CONFIG
from llm.ollama_client import chat, _apply_rate_limit
from llm.explanation import (
    _involved_templates, _templates_blob, _scrub_narrative, _term_in_templates,
)

log = logging.getLogger(__name__)

# Security/threat terms the professor explicitly forbids unless supported by the
# logs. These EXTEND (do not replace) llm.explanation._BANNED_CAUSE_TERMS, which
# already covers attack / malicious / intrusion / network / hardware / failure but
# NOT these. A sentence containing any of these is dropped unless the term is
# present verbatim in the involved templates.
SECURITY_TERMS = (
    "malware", "ransomware", "spyware", "trojan", "virus", "worm",
    "insider", "threat", "cyberattack", "cyber-attack", "cyber attack",
    "breach", "exploit", "phishing", "ddos", "espionage", "hacker", "hacking",
    "backdoor", "rootkit", "exfiltration", "compromise", "adversary",
)

# Template tokens that, when present in the ACTUAL event, constitute grounded
# evidence of a problem (used by the deterministic baseline and to guide the prompt).
FAILURE_TOKENS = ("exception", "error", "failed", "failure", "timeout",
                  "denied", "interrupted", "abort")

# Exact wording the professor requires when the evidence does not support a call.
INSUFFICIENT_EVIDENCE = (
    "There is insufficient evidence in the provided logs to determine the root cause."
)

# Hedging words a FINAL decision must never contain (professor: no "Maybe"/"Possibly").
_HEDGE_WORDS = ("maybe", "possibly", "perhaps", "probably", "might", "could be",
                "uncertain", "not sure", "unsure", "likely")

_SYSTEM_PROMPT = (
    "You are a log-analysis assistant for HDFS. You are the SECOND-STAGE decision "
    "maker: the GRU detector was UNSURE about this one operation, so you must give a "
    "FINAL verdict of exactly NORMAL or SUSPICIOUS, using ONLY the facts in the "
    "BEGIN_FACTS/END_FACTS block. Never output 'maybe', 'possibly', or any hedging "
    "word. You must never invent causes. Do NOT mention malware, attacks, intrusions, "
    "hardware, disks, memory, networks, or configuration unless that exact word "
    "appears in the templates shown. Rules for the decision:\n"
    "  - SUSPICIOUS only if the ACTUAL event's template itself records a failure, "
    "exception, error, interruption, or denial.\n"
    "  - NORMAL otherwise. An operation may look unusual or out-of-order yet still "
    "be NORMAL — unusual ordering ALONE is not suspicious; only an actual "
    "failure/exception in the actual event is. When in doubt, answer NORMAL.\n"
    "  - ONLY if the actual event's template genuinely shows nothing to judge on, "
    "make the EXPLANATION exactly: " + INSUFFICIENT_EVIDENCE + "\n"
    "Answer in EXACTLY these four lines and nothing else:\n"
    "CLASSIFICATION: <NORMAL or SUSPICIOUS>\n"
    "CONFIDENCE: <integer 0-100, e.g. 75>\n"
    "EVIDENCE: <one short sentence quoting words from the ACTUAL event's template above>\n"
    "EXPLANATION: <one factual sentence stating what the ACTUAL event's template "
    "records and whether it fits the sequence so far; no speculation, no invented causes>"
)

_USER_TEMPLATE = """\
BEGIN_FACTS
Observed sequence ({n_events} events, in order):
{sequence_with_templates}

Expected next event : {pred_event} -> {pred_template}
Actual next event   : {actual_event} -> {actual_template}
Anomaly score (0-1, higher = more surprising) : {anomaly_score:.2f}
Confidence score (0-1, model's prob of the actual event) : {confidence:.2f}
Routing reason : {routing_reason}
END_FACTS

Give the FINAL classification (NORMAL or SUSPICIOUS) for this operation using ONLY \
the facts above."""


def _field(label, text):
    """Extract a single LINE field like 'LABEL: value' (robust to a 1B model's noise)."""
    m = re.search(rf"{label}\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_classification(text):
    """Return 'SUSPICIOUS' / 'NORMAL' / None (unparseable) from the model text."""
    raw = _field("CLASSIFICATION", text).upper()
    has_susp = "SUSPICIOUS" in raw or "ANOMAL" in raw
    has_norm = "NORMAL" in raw
    if has_susp and not has_norm:
        return "SUSPICIOUS"
    if has_norm and not has_susp:
        return "NORMAL"
    # Fallback: scan the whole response if the label line was malformed.
    up = text.upper()
    if "SUSPICIOUS" in up and "NORMAL" not in up:
        return "SUSPICIOUS"
    if "NORMAL" in up and "SUSPICIOUS" not in up:
        return "NORMAL"
    return None


def _parse_confidence(text):
    m = re.search(r"CONFIDENCE\s*:\s*(\d{1,3})", text, flags=re.IGNORECASE)
    if not m:
        return None
    return max(0.0, min(1.0, int(m.group(1)) / 100.0))


def _scrub_security(text, blob):
    """Drop any sentence naming a banned security term not present in the templates.
    Strengthens (never weakens) the production scrubbers."""
    if not text:
        return ""
    kept = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        low = s.lower()
        bad = [t for t in SECURITY_TERMS
               if t in low and not _term_in_templates(t.split()[0], blob)]
        if bad:
            continue
        if s.strip():
            kept.append(s.strip())
    return " ".join(kept).strip()


def _has_unsupported_banned(text, blob):
    """True if the RAW text names any security term not supported by the templates
    (used to measure the raw hallucination rate, before scrubbing)."""
    low = (text or "").lower()
    return any(t in low and not _term_in_templates(t.split()[0], blob)
               for t in SECURITY_TERMS)


def classify_uncertain(sequence, predicted_event, actual_event, event_context,
                       confidence, anomaly_score, routing_reason="",
                       temperature=0.0, model_name=None):
    """Ask Llama for a FINAL constrained NORMAL/SUSPICIOUS verdict on one uncertain case.

    The prompt is grounded ONLY in: observed sequence, expected vs actual event and
    their templates, the anomaly score, the confidence score, and the routing reason.

    Returns a dict: classification, confidence, evidence, explanation, raw,
    parse_ok, hallucinated (raw named an unsupported banned term), scrubbed
    (text changed during scrubbing), insufficient (fell back to the exact
    insufficient-evidence sentence), latency (seconds), error.
    """
    involved = _involved_templates(sequence, predicted_event, actual_event, event_context)
    blob = _templates_blob(involved)
    seq_lines = "\n".join(
        f"  {i:2d}. {e} -> {involved.get(e) or 'template unavailable'}"
        for i, e in enumerate(sequence, 1))
    user = _USER_TEMPLATE.format(
        n_events=len(sequence), sequence_with_templates=seq_lines,
        pred_event=predicted_event, pred_template=involved.get(predicted_event) or "n/a",
        actual_event=actual_event, actual_template=involved.get(actual_event) or "n/a",
        anomaly_score=anomaly_score, confidence=confidence,
        routing_reason=routing_reason or "GRU detector was not confident (UNCERTAIN band).")

    model_name = model_name or CONFIG.get("ollama_model", "llama3.2:1b")
    out = {"classification": None, "confidence": None, "evidence": "",
           "explanation": "", "raw": "", "parse_ok": False, "hallucinated": False,
           "scrubbed": False, "insufficient": False, "latency": 0.0, "error": None}

    if chat is None:
        out["error"] = "ollama-not-installed"
        return out

    _apply_rate_limit(CONFIG.get("ollama_rate_limit_seconds", 0.5))
    t0 = time.perf_counter()
    try:
        resp = chat(model=model_name,
                    messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                              {"role": "user", "content": user}],
                    options={"temperature": temperature})
        raw = (resp.get("message", {}) or {}).get("content", "") or str(resp)
    except Exception as exc:  # network / model error — report, never fabricate
        out["latency"] = time.perf_counter() - t0
        out["error"] = repr(exc)
        return out
    out["latency"] = time.perf_counter() - t0
    out["raw"] = raw.strip()

    # --- Parse the four line-fields; force a binary FINAL label (no hedging) ---
    cls = _parse_classification(raw)
    out["parse_ok"] = cls is not None
    out["classification"] = cls or "NORMAL"   # conservative default when unparseable
    out["confidence"] = _parse_confidence(raw)

    ev_raw = _field("EVIDENCE", raw)
    ex_raw = _field("EXPLANATION", raw)

    # --- Hallucination flag (raw) then scrub (production + security) ---
    out["hallucinated"] = (_has_unsupported_banned(ev_raw, blob)
                           or _has_unsupported_banned(ex_raw, blob))
    ev_clean = _scrub_security(_scrub_narrative(ev_raw, blob), blob)
    ex_clean = _scrub_security(_scrub_narrative(ex_raw, blob), blob)
    out["scrubbed"] = (ev_clean != ev_raw.strip()) or (ex_clean != ex_raw.strip())

    # --- Insufficient-evidence guarantee (professor Req 3, deterministic) ---
    # If the model already said so, or scrubbing left no usable explanation, emit the
    # exact required sentence rather than guessing or returning empty text.
    said_insufficient = "insufficient evidence" in (ex_raw + " " + ev_raw).lower()
    if said_insufficient or not ex_clean:
        ex_clean = INSUFFICIENT_EVIDENCE
        out["insufficient"] = True

    out["evidence"] = ev_clean
    out["explanation"] = ex_clean
    return out


# =============================================================================
# v2 classifier — Task 4 improvement ladder. The production classify_uncertain()
# above is UNTOUCHED (tests, --demo-uncertain, and the detection pipeline keep
# their exact behaviour); v2 is opt-in, with each improvement individually
# togglable so the evaluation ladder can isolate what helps:
#   * dataset-aware framing        (v1 hardcodes "for HDFS" even on BGL data)
#   * richer grounded evidence     (block-surprise percentile vs normal validation
#                                   blocks; template frequency in normal training
#                                   data; top-3 most-surprising windows — cheap
#                                   retrieval from data the detector already has)
#   * relaxed SUSPICIOUS criteria  (v1's failure-token-only rule structurally
#                                   yields ~0% recall on the uncertain band —
#                                   uncertain-band anomalies rarely contain
#                                   failure tokens; v2 adds never-seen-in-normal
#                                   and extreme-percentile as grounded reasons)
#   * few-shot exemplars           (labelled cases from the VALIDATION uncertain
#                                   band — never test)
#   * structured outputs           (Ollama JSON-schema constrained decoding, with
#                                   the v1 line-format parser as fallback)
#   * self-consistency             (k samples at temperature>0, majority vote)
# All v2 prose flows through the SAME scrubbers as v1 — safety is unchanged.
# =============================================================================

_DATASET_FRAMING = {
    "hdfs": ("HDFS (Hadoop Distributed File System) datanode logs; one case is a "
             "single HDFS operation's event sequence"),
    "bgl": ("BGL (Blue Gene/L supercomputer) RAS logs; one case is a fixed window "
            "of consecutive log lines from the machine"),
}

_V2_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["NORMAL", "SUSPICIOUS"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "evidence": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["classification", "confidence", "evidence", "explanation"],
}

_V2_STRICT_RULES = (
    "  - SUSPICIOUS only if the ACTUAL event's template itself records a failure, "
    "exception, error, interruption, or denial.\n"
    "  - NORMAL otherwise. An operation may look unusual or out-of-order yet still "
    "be NORMAL — unusual ordering ALONE is not suspicious. When in doubt, answer "
    "NORMAL.\n")

_V2_RELAXED_RULES = (
    "  - SUSPICIOUS if ANY of these grounded conditions holds:\n"
    "      (a) the ACTUAL event's template records a failure, exception, error, "
    "interruption, or denial;\n"
    "      (b) the FACTS state the actual event was NEVER seen (or seen only a "
    "handful of times) in normal training data;\n"
    "      (c) the FACTS state this block's surprise percentile among normal "
    "blocks is 99 or higher.\n"
    "  - NORMAL otherwise: a common template with a low surprise percentile is "
    "NORMAL even if the ordering looks unusual. If none of (a)/(b)/(c) holds, "
    "answer NORMAL.\n")


def _v2_system_prompt(dataset, relaxed, structured):
    framing = _DATASET_FRAMING.get(dataset, _DATASET_FRAMING["hdfs"])
    rules = _V2_RELAXED_RULES if relaxed else _V2_STRICT_RULES
    if structured:
        answer_spec = (
            "Answer with a single JSON object with exactly these keys:\n"
            '{"classification": "NORMAL or SUSPICIOUS", "confidence": <0-100>, '
            '"evidence": "<one short sentence quoting words from the FACTS>", '
            '"explanation": "<one factual sentence grounded in the FACTS; no '
            'speculation, no invented causes>"}')
    else:
        answer_spec = (
            "Answer in EXACTLY these four lines and nothing else:\n"
            "CLASSIFICATION: <NORMAL or SUSPICIOUS>\n"
            "CONFIDENCE: <integer 0-100, e.g. 75>\n"
            "EVIDENCE: <one short sentence quoting words from the FACTS above>\n"
            "EXPLANATION: <one factual sentence grounded in the FACTS; no "
            "speculation, no invented causes>")
    return (
        f"You are a log-analysis assistant for {framing}. You are the SECOND-STAGE "
        "decision maker: the GRU detector was UNSURE about this one case, so you "
        "must give a FINAL verdict of exactly NORMAL or SUSPICIOUS, using ONLY the "
        "facts in the BEGIN_FACTS/END_FACTS block. Never output 'maybe', "
        "'possibly', or any hedging word. You must never invent causes. Do NOT "
        "mention malware, attacks, intrusions, hardware, disks, memory, networks, "
        "or configuration unless that exact word appears in the templates shown. "
        "Rules for the decision:\n" + rules +
        "  - ONLY if the facts genuinely show nothing to judge on, make the "
        "explanation exactly: " + INSUFFICIENT_EVIDENCE + "\n" + answer_spec)


def build_user_v2(sequence, predicted_event, actual_event, event_context,
                  confidence, anomaly_score, routing_reason="", evidence=None):
    """Build the v2 BEGIN_FACTS user message. `evidence` (optional) is the cheap
    retrieval dict: {score_percentile, actual_freq, total_events, pred_freq,
    top_windows: [{rank, actual_event, actual_template, surprisal}]}."""
    involved = _involved_templates(sequence, predicted_event, actual_event, event_context)
    seq_lines = "\n".join(
        f"  {i:2d}. {e} -> {involved.get(e) or 'template unavailable'}"
        for i, e in enumerate(sequence, 1))
    lines = [
        "BEGIN_FACTS",
        f"Observed sequence ({len(sequence)} events, in order):",
        seq_lines,
        "",
        f"Expected next event : {predicted_event} -> {involved.get(predicted_event) or 'n/a'}",
        f"Actual next event   : {actual_event} -> {involved.get(actual_event) or 'n/a'}",
        f"Anomaly score (0-1, higher = more surprising) : {anomaly_score:.2f}",
        f"Confidence score (0-1, model's prob of the actual event) : {confidence:.2f}",
    ]
    if evidence:
        if evidence.get("score_percentile") is not None:
            lines.append(
                f"Block surprise percentile among NORMAL validation blocks : "
                f"{evidence['score_percentile']:.1f} (100 = more surprising than every "
                "normal block)")
        if evidence.get("actual_freq") is not None:
            freq = evidence["actual_freq"]; tot = evidence.get("total_events", 0)
            desc = ("NEVER seen" if freq == 0 else f"seen {freq:,} times")
            lines.append(
                f"Actual event {actual_event} in NORMAL training data : {desc} "
                f"among {tot:,} events")
        if evidence.get("pred_freq") is not None:
            lines.append(
                f"Expected event {predicted_event} in NORMAL training data : seen "
                f"{evidence['pred_freq']:,} times")
        for w in (evidence.get("top_windows") or []):
            lines.append(
                f"Surprising transition #{w['rank']} in this block : actual "
                f"{w['actual_event']} -> {w['actual_template']} (surprise "
                f"{w['surprisal']:.2f})")
    lines += [
        f"Routing reason : {routing_reason or 'GRU detector was not confident (UNCERTAIN band).'}",
        "END_FACTS",
        "",
        "Give the FINAL classification (NORMAL or SUSPICIOUS) for this case using "
        "ONLY the facts above.",
    ]
    return "\n".join(lines)


def _parse_v2_json(raw):
    """Parse the structured-output JSON; returns (cls, conf, evidence, explanation)
    or None if the text is not valid JSON of the expected shape."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    cls = str(obj.get("classification", "")).upper().strip()
    if cls not in ("NORMAL", "SUSPICIOUS"):
        return None
    conf = obj.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf) / 100.0))
    except (TypeError, ValueError):
        conf = None
    return cls, conf, str(obj.get("evidence", "")), str(obj.get("explanation", ""))


def classify_uncertain_v2(sequence, predicted_event, actual_event, event_context,
                          confidence, anomaly_score, routing_reason="",
                          temperature=0.0, model_name=None, dataset="hdfs",
                          evidence=None, few_shot=None, relaxed=False,
                          structured=False, self_consistency_k=1):
    """Improved uncertain-band classifier (see module banner). Same return shape
    as classify_uncertain, plus 'votes' when self_consistency_k > 1."""
    involved = _involved_templates(sequence, predicted_event, actual_event, event_context)
    blob = _templates_blob(involved)
    user = build_user_v2(sequence, predicted_event, actual_event, event_context,
                         confidence, anomaly_score, routing_reason, evidence)

    messages = [{"role": "system",
                 "content": _v2_system_prompt(dataset, relaxed, structured)}]
    for ex_user, ex_answer in (few_shot or []):
        messages.append({"role": "user", "content": ex_user})
        messages.append({"role": "assistant", "content": ex_answer})
    messages.append({"role": "user", "content": user})

    model_name = model_name or CONFIG.get("ollama_model", "llama3.2:1b")
    out = {"classification": None, "confidence": None, "evidence": "",
           "explanation": "", "raw": "", "parse_ok": False, "hallucinated": False,
           "scrubbed": False, "insufficient": False, "latency": 0.0, "error": None,
           "votes": []}
    if chat is None:
        out["error"] = "ollama-not-installed"
        return out

    k = max(1, int(self_consistency_k))
    # A deterministic single call stays at the caller's temperature; a majority
    # vote needs diversity, so k>1 forces temperature>0.
    temp = temperature if k == 1 else max(temperature, 0.7)
    responses = []
    for _ in range(k):
        _apply_rate_limit(CONFIG.get("ollama_rate_limit_seconds", 0.5))
        t0 = time.perf_counter()
        try:
            kwargs = {"model": model_name, "messages": messages,
                      "options": {"temperature": temp}}
            if structured:
                kwargs["format"] = _V2_JSON_SCHEMA
            resp = chat(**kwargs)
            raw = (resp.get("message", {}) or {}).get("content", "") or str(resp)
        except Exception as exc:
            out["latency"] += time.perf_counter() - t0
            out["error"] = repr(exc)
            if not responses:
                return out
            break
        out["latency"] += time.perf_counter() - t0
        responses.append(raw.strip())

    # Parse every response; majority vote on the label.
    parsed = []
    for raw in responses:
        got = _parse_v2_json(raw) if structured else None
        if got is None:
            cls = _parse_classification(raw)
            conf = _parse_confidence(raw)
            ev_raw, ex_raw = _field("EVIDENCE", raw), _field("EXPLANATION", raw)
            parsed.append({"cls": cls, "conf": conf, "ev": ev_raw, "ex": ex_raw,
                           "raw": raw, "json_ok": False})
        else:
            cls, conf, ev_raw, ex_raw = got
            parsed.append({"cls": cls, "conf": conf, "ev": ev_raw, "ex": ex_raw,
                           "raw": raw, "json_ok": True})

    votes = [p["cls"] for p in parsed if p["cls"] is not None]
    out["votes"] = votes
    if votes:
        n_susp = sum(1 for v in votes if v == "SUSPICIOUS")
        majority = "SUSPICIOUS" if n_susp * 2 > len(votes) else "NORMAL"
    else:
        majority = None
    # Representative response = first one that voted with the majority.
    rep = next((p for p in parsed if p["cls"] == majority), parsed[0] if parsed else None)

    out["raw"] = "\n---\n".join(responses)
    out["parse_ok"] = majority is not None
    out["classification"] = majority or "NORMAL"   # conservative default (as v1)
    out["confidence"] = rep["conf"] if rep else None

    ev_raw = rep["ev"] if rep else ""
    ex_raw = rep["ex"] if rep else ""
    out["hallucinated"] = (_has_unsupported_banned(ev_raw, blob)
                           or _has_unsupported_banned(ex_raw, blob))
    ev_clean = _scrub_security(_scrub_narrative(ev_raw, blob), blob)
    ex_clean = _scrub_security(_scrub_narrative(ex_raw, blob), blob)
    out["scrubbed"] = (ev_clean != ev_raw.strip()) or (ex_clean != ex_raw.strip())

    said_insufficient = "insufficient evidence" in (ex_raw + " " + ev_raw).lower()
    if said_insufficient or not ex_clean:
        ex_clean = INSUFFICIENT_EVIDENCE
        out["insufficient"] = True

    out["evidence"] = ev_clean
    out["explanation"] = ex_clean
    return out

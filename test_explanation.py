# =============================================================================
# tests/test_explanation.py — Regression suite for the evidence-based
# explanation layer (llm/explanation.py).
#
# Plain-assert tests, no pytest dependency. Run from the repo root:
#     venv/bin/python tests/test_explanation.py
#
# Verifies the 7 explanation-hardening requirements: verdict scrubbing,
# banned-cause scrubbing (single + multi-word + template exception),
# speculation scrubbing, the root-cause two-case rule, single-section
# parsing, and deterministic report assembly.
# =============================================================================

import os
import re
import sys

# Allow `from llm.explanation import ...` when executed as a script (the script's
# own directory, not the repo root, is sys.path[0] by default).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.explanation import (          # noqa: E402
    _scrub_verdict_phrases,
    _scrub_speculation,
    _scrub_contradiction,
    _scrub_narrative,
    _templates_blob,
    _root_cause_assessment,
    _parse_narrative_sections,
    _assemble_evidence_report,
    _deviation_analysis,
    _confidence_analysis,
    generate_explanation,
    _INSUFFICIENT_EVIDENCE,
    _FAILURE_TEMPLATE_TOKENS,
    _BANNED_CAUSE_TERMS,
)


def test_verdict_phrases_scrubbed():
    """Req 3 — sentences echoing verdict fields are dropped; others survive."""
    result = _scrub_verdict_phrases(
        "The confidence score is 0.9123. Event logged successfully.")
    assert "confidence score" not in result.lower(), (
        f"verdict phrase not removed; got: {result!r}")
    assert "Event logged successfully." in result, (
        f"non-verdict sentence wrongly removed; got: {result!r}")

    result2 = _scrub_verdict_phrases(
        "Prediction correct: True. The block transition completed.")
    assert "prediction correct" not in result2.lower(), (
        f"'prediction correct' not removed; got: {result2!r}")


def test_banned_single_word_scrubbed():
    """Req 4 — a banned cause not in the templates removes the whole sentence."""
    clean_blob = _templates_blob({"E1": "block received"})
    scrubbed = _scrub_narrative("Disk failure detected.", clean_blob)
    assert scrubbed == "", f"'disk' sentence not removed; got: {scrubbed!r}"


def test_banned_multi_word_scrubbed():
    """Req 4 — multi-word banned phrases are removed too."""
    clean_blob = _templates_blob({"E1": "block received"})
    scrubbed = _scrub_narrative("This is a resource constraint.", clean_blob)
    assert "resource constraint" not in scrubbed.lower(), (
        f"multi-word phrase not removed; got: {scrubbed!r}")


def test_banned_term_in_template_is_kept():
    """Req 4 — a banned term IS allowed when it appears in an involved template."""
    disk_blob = _templates_blob({"E1": "disk write error detected"})
    kept = _scrub_narrative("Disk write was attempted.", disk_blob)
    assert "disk" in kept.lower(), (
        f"'disk' sentence removed even though disk is in templates; got: {kept!r}")


def test_speculation_scrubbed():
    """Req 5 — speculation / causal-inference language is dropped."""
    spec1 = _scrub_speculation(
        "This likely indicates a problem. Event logged successfully.")
    assert "likely" not in spec1.lower(), f"'likely' not removed; got: {spec1!r}"
    assert "Event logged successfully." in spec1, (
        f"non-speculation sentence wrongly removed; got: {spec1!r}")

    spec2 = _scrub_speculation("The event suggests a failure.")
    assert spec2 == "", f"'suggests' sentence not removed; got: {spec2!r}"


def test_root_cause_two_case_rule():
    """Req 6 — failure token in the actual template is named; otherwise standard."""
    hit_inv = {"E5": "java.io.IOException: could not read block (error 5)"}
    rc_hit = _root_cause_assessment("E5", hit_inv)
    assert any(tok in rc_hit for tok in _FAILURE_TEMPLATE_TOKENS), (
        f"failure token not named in root cause; got: {rc_hit!r}")
    assert "deeper root cause" in rc_hit, (
        f"expected 'deeper root cause' phrase; got: {rc_hit!r}")

    no_hit_inv = {"E5": "Receiving block from datanode"}
    rc_no_hit = _root_cause_assessment("E5", no_hit_inv)
    assert rc_no_hit == _INSUFFICIENT_EVIDENCE, (
        f"expected _INSUFFICIENT_EVIDENCE verbatim; got: {rc_no_hit!r}")


def test_parser_returns_only_interpretation():
    """Req 2 — only EVIDENCE_BASED_INTERPRETATION is parsed; other sections dropped."""
    raw_llm = ("OBSERVED_SEQUENCE_PATTERN:\nsome pattern text\n"
               "EVIDENCE_BASED_INTERPRETATION:\nthe real interpretation")
    parsed = _parse_narrative_sections(raw_llm)
    assert "the real interpretation" in parsed, (
        f"EVIDENCE_BASED_INTERPRETATION not parsed; got: {parsed!r}")
    assert "some pattern text" not in parsed, (
        f"OBSERVED_SEQUENCE_PATTERN leaked into parsed; got: {parsed!r}")


def test_report_is_deterministic():
    """Req 1 — same kwargs → identical report; all Python-owned sections present."""
    inv = {"E6": "Receiving block", "E11": "Served block to client"}
    kwargs = dict(
        sequence=["E6", "E6"], predicted_event="E6", actual_event="E11",
        confidence=0.8123, anomaly_score=0.7456, classification="ANOMALY",
        prediction_correct=False, involved=inv,
        interpretation="E11 was observed instead of E6.",
    )
    r1 = _assemble_evidence_report(**kwargs)
    r2 = _assemble_evidence_report(**kwargs)
    assert r1 == r2, "_assemble_evidence_report is not deterministic"

    # Existing required sections + the new Deviation Analysis / Confidence Analysis.
    for section in ("Summary", "Prediction", "Confidence",
                    "Root Cause Assessment", "Recommended Investigation",
                    "Deviation Analysis", "Interpretation"):
        assert section in r1, f"required section '{section}' missing from report"


def test_deviation_analysis_is_clean_and_grounded():
    """New section — template-grounded, deterministic, and contains no banned cause
    that is not present in the templates (so it can never hallucinate a cause)."""
    inv = {"E5": "Receiving block from datanode", "E11": "PacketResponder terminating"}
    blob = _templates_blob(inv)

    # Deterministic.
    d1 = _deviation_analysis("E5", "E11", False, inv)
    d2 = _deviation_analysis("E5", "E11", False, inv)
    assert d1 == d2, "_deviation_analysis is not deterministic"

    # Idempotent under the scrubber == carries no unsupported banned cause / speculation.
    assert _scrub_narrative(d1, blob).strip() == d1.strip(), (
        f"deviation analysis would be scrubbed (unsupported term?); got: {d1!r}")
    assert "E5" in d1 and "E11" in d1, "deviation analysis should name both events"

    # The correct-prediction branch states the order matched.
    d_match = _deviation_analysis("E5", "E5", True, inv)
    assert "matches" in d_match.lower(), f"expected 'matches' wording; got: {d_match!r}"


def test_confidence_analysis_is_clean():
    """Task 6 — plain-language, percentage-formatted, deterministic, no banned cause.

    The interpretation is written for a non-technical reader: it shows the
    confidence as a percentage (no raw 4-decimal numbers) and says what the
    detector did. It must still never contain a hallucinated root-cause term."""
    c = _confidence_analysis(0.495, 0.505, "UNCERTAIN")
    assert c == _confidence_analysis(0.495, 0.505, "UNCERTAIN"), "not deterministic"
    assert "49.5%" in c, f"confidence should render as a percentage; got: {c!r}"
    assert "0.49" not in c, f"raw decimal leaked into plain-language text; got: {c!r}"
    assert "review" in c.lower(), f"UNCERTAIN case should mention review; got: {c!r}"
    low = c.lower()
    for term in _BANNED_CAUSE_TERMS:
        assert not re.search(rf"\b{re.escape(term)}\b", low), (
            f"confidence analysis contains banned cause term {term!r}; got: {c!r}")


def test_no_deviation_sentence_when_prediction_correct():
    """Task 5 — when expected == actual, Deviation Analysis states it explicitly and
    never implies a change occurred."""
    inv = {"E5": "Receiving block from datanode"}
    d = _deviation_analysis("E5", "E5", True, inv)
    assert "No workflow deviation was detected." in d, (
        f"required no-deviation sentence missing; got: {d!r}")
    for bad in ("instead of", "different event", "deviates"):
        assert bad not in d.lower(), (
            f"correct-prediction text implies a change ({bad!r}); got: {d!r}")


def test_contradiction_scrubber():
    """Task 5 — deviation-implying sentences are dropped ONLY when the prediction
    is correct; for a real deviation the same language is accurate and kept."""
    txt = "The actual event E11 occurred instead of E6. This is a clear deviation."
    assert _scrub_contradiction(txt, True).strip() == "", (
        f"contradiction not scrubbed when prediction correct; got: "
        f"{_scrub_contradiction(txt, True)!r}")
    assert _scrub_contradiction(txt, False) == txt, (
        "scrubber must be a no-op for genuine deviations")


def test_correct_report_never_contradicts():
    """Task 5 (end-to-end) — a correct-prediction report carries the no-deviation
    sentence, strips a contradictory LLM-style interpretation, and shows the
    confidence as a percentage (Task 6)."""
    inv = {"E9": "PacketResponder for block terminating"}
    report = _assemble_evidence_report(
        sequence=["E9", "E9"], predicted_event="E9", actual_event="E9",
        confidence=0.495, anomaly_score=0.505, classification="UNCERTAIN",
        prediction_correct=True, involved=inv,
        interpretation="The event deviates from the expected workflow, occurring instead of E9.",
    )
    assert "No workflow deviation was detected." in report, "no-deviation sentence missing"
    assert "deviates" not in report.lower(), "contradictory narrative leaked into report"
    assert "instead of" not in report.lower(), "contradictory narrative leaked into report"
    assert "49.5%" in report, "confidence should be shown as a percentage"


def test_generate_explanation_correct_prediction_is_safe_offline():
    """Task 4 (public path) — the FULL generate_explanation entry point, with
    expected == actual, must never contradict the deterministic facts. Because the
    LLM is structurally skipped when the prediction is correct, this runs offline
    (no Ollama) and proves the guarantee holds end-to-end, not just in the assembler.
    """
    inv = {"E9": "PacketResponder for block terminating"}
    report = generate_explanation(
        sequence=["E9", "E9"], predicted_event="E9", actual_event="E9",
        confidence=0.49, anomaly_score=0.51, classification="UNCERTAIN",
        event_context=inv, use_llm=True,    # requested, but skipped because prediction is correct
    )
    assert "Prediction Correct: True" in report, f"expected correct verdict; got:\n{report}"
    assert "No workflow deviation was detected." in report, "no-deviation sentence missing"
    for bad in ("instead of", "deviates", "different event", "does not match", "mismatch"):
        assert bad not in report.lower(), f"contradiction phrase {bad!r} leaked into report"


# Ordered list of every test in the suite.
_ALL_TESTS = [
    test_verdict_phrases_scrubbed,
    test_banned_single_word_scrubbed,
    test_banned_multi_word_scrubbed,
    test_banned_term_in_template_is_kept,
    test_speculation_scrubbed,
    test_root_cause_two_case_rule,
    test_parser_returns_only_interpretation,
    test_report_is_deterministic,
    test_deviation_analysis_is_clean_and_grounded,
    test_confidence_analysis_is_clean,
    test_no_deviation_sentence_when_prediction_correct,
    test_contradiction_scrubber,
    test_correct_report_never_contradicts,
    test_generate_explanation_correct_prediction_is_safe_offline,
]


def run_verification_tests() -> None:
    """Run every assertion in the suite. Raises AssertionError on first failure.

    Kept as a single callable so it can be invoked from a harness or REPL:
        venv/bin/python -c "import tests.test_explanation as t; t.run_verification_tests()"
    """
    for test in _ALL_TESTS:
        test()


if __name__ == "__main__":
    passed = 0
    for test in _ALL_TESTS:
        test()                      # raises AssertionError with context on failure
        print(f"  PASS  {test.__name__}")
        passed += 1
    print(f"\nAll {passed} explanation-layer tests passed.")

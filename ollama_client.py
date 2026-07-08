# =============================================================================
# llm/ollama_client.py — Low-level Ollama access + shared LLM state.
#
# Owns the single source of truth for: the `chat` handle (None if Ollama is
# absent), thread-safe call statistics, the explanation response cache, the
# rate limiter, and the connectivity probe. `llm/explanation.py` imports these;
# this module imports nothing from explanation — keeping the package acyclic.
# =============================================================================

import time
import hashlib
import logging
import threading
from typing import Dict, List

from config import CONFIG

log = logging.getLogger(__name__)

# The Ollama Python client. Stays None when the package is not installed, which
# lets every caller degrade gracefully to deterministic, template-only output.
try:
    from ollama import chat
except Exception:
    chat = None


# ---------------------------------------------------------------------------
# LLM Statistics — counters for total anomalies seen, LLM calls made, cache
# hits, failed calls, and cumulative call latency. Thread-safe via a lock so
# concurrent callers don't race.
#   - total_failures        : how many chat() attempts raised an exception.
#   - total_latency_seconds : summed wall-clock time of chat() attempts; divide
#                             by total_llm_calls for the average latency.
# (Dict[str, float] now that latency is tracked — the integer counters are still
#  whole numbers, but the dict holds a float total.)
# ---------------------------------------------------------------------------
_llm_stats_lock = threading.Lock()
_llm_stats: Dict[str, float] = {
    "total_anomalies":       0,
    "total_llm_calls":       0,
    "total_cached":          0,
    "total_failures":        0,
    "total_latency_seconds": 0.0,
}

# ---------------------------------------------------------------------------
# Explanation cache — maps a fingerprint of (sequence, pred, actual, scores)
# to the previously generated explanation string, avoiding redundant Ollama
# calls for identical anomaly patterns.
# ---------------------------------------------------------------------------
_explanation_cache: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Rate limiter — enforces a minimum gap between consecutive Ollama calls so
# the local model is not overwhelmed. Uses a simple last-call timestamp with
# a configurable minimum interval (seconds).
# ---------------------------------------------------------------------------
_rate_limiter_lock = threading.Lock()
_rate_limiter_last_call: float = 0.0


def _make_cache_key(sequence: List[str], predicted_event: str, actual_event: str,
                    confidence: float, anomaly_score: float,
                    classification: str) -> str:
    """Deterministic fingerprint for an anomaly case, used as the cache key."""
    raw = (f"{sequence}|{predicted_event}|{actual_event}"
           f"|{confidence:.4f}|{anomaly_score:.4f}|{classification}")
    return hashlib.sha256(raw.encode()).hexdigest()


def _apply_rate_limit(min_interval: float) -> None:
    """Blocks the calling thread until the rate-limit interval has elapsed."""
    with _rate_limiter_lock:
        global _rate_limiter_last_call
        elapsed = time.monotonic() - _rate_limiter_last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _rate_limiter_last_call = time.monotonic()


def get_llm_stats() -> Dict[str, int]:
    """Returns a snapshot copy of the current LLM call statistics."""
    with _llm_stats_lock:
        return dict(_llm_stats)


def log_llm_stats() -> None:
    """Logs the current LLM call statistics at INFO level."""
    stats = get_llm_stats()
    calls = stats["total_llm_calls"]
    avg_latency = stats["total_latency_seconds"] / calls if calls else 0.0
    log.info(
        "LLM stats — total_anomalies=%s  total_llm_calls=%s  total_cached=%s  "
        "total_failures=%s  avg_latency=%.2fs",
        int(stats["total_anomalies"]),
        int(stats["total_llm_calls"]),
        int(stats["total_cached"]),
        int(stats["total_failures"]),
        avg_latency,
    )


def test_ollama_connection(model_name: str = None,
                           retries: int = None,
                           backoff: float = None):
    """
    Performs a real `chat()` call to verify Ollama and model availability.
    Returns (ok: bool, model_ok: bool, message: str).
    """
    if model_name is None:
        model_name = CONFIG.get("ollama_model", "llama3.2:1b")
    if retries is None:
        retries = CONFIG.get("ollama_retries", 2)
    if backoff is None:
        backoff = CONFIG.get("ollama_backoff", 1.5)

    if chat is None:
        return False, False, "Ollama Python client not installed"

    last_exc = None
    system_prompt = "You are a connectivity test. Reply with 'pong'."
    user_prompt = "Please reply with the single word: pong"

    for attempt in range(1, retries + 1):
        try:
            resp = chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            try:
                content = resp["message"]["content"]
                if "pong" in content.lower():
                    return True, True, "OK"
                return True, True, "Model responded"
            except Exception:
                return True, True, "Model responded (unknown format)"

        except Exception as exc:
            last_exc = exc
            log.debug("Ollama chat probe attempt %s/%s failed: %s", attempt, retries, repr(exc))
            if attempt < retries:
                time.sleep(backoff ** attempt)

    return False, False, f"Ollama chat probe failed: {last_exc}"

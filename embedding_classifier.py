# =============================================================================
# llm/embedding_classifier.py — PRODUCTION Uncertain-band classifier:
# Llama EMBEDDINGS + a trained logistic head (representation probing).
#
# What is embedded: the block's ordered-unique event-template TEXT
# (`block_text`) — semantic log content only. No labels, no engineered
# features, so the head generalizes across datasets where the deterministic
# never-seen rule does not (it has 0 recall on HDFS).
#
# Fallback contract (a decision is NEVER fabricated): if the head artifact is
# missing/corrupt, Ollama is unreachable, embedding fails, or a block is
# invalid, `resolve_uncertain` returns verdict=None with a fallback reason and
# the caller keeps the block UNCERTAIN.
#
# Artifacts: outputs/uncertain_sets/uncertain_head_{hdfs,bgl}.joblib — written
# by train_uncertain_head.py with FROZEN, dev-selected configs (BGL LogReg
# C=0.1 thr 0.598; HDFS LogReg C=0.01 thr 0.76). The trainer hard-asserts the
# published held-out confusions reproduce before saving.
# =============================================================================

import json
import logging
import urllib.request
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

EMBED_URL = "http://localhost:11434/api/embed"
FALLBACK_HEAD_MISSING = "head-missing"
FALLBACK_OLLAMA_DOWN = "ollama-down"
FALLBACK_EMBED_ERROR = "embed-error"
FALLBACK_INVALID_INPUT = "invalid-input"


def block_text(block, encoder, event_map) -> str:
    """Ordered UNIQUE event-template texts of a block — the semantic log
    content the head classifies. Deliberately label-free and feature-free."""
    names = encoder.inverse_transform(list(block), context="emb")
    seen, parts = set(), []
    for e in names:
        if e in seen:
            continue
        seen.add(e)
        parts.append(event_map.get(e, e) or e)
    return " | ".join(parts)


def embed_texts(texts: List[str], model: str, batch: int = 16, timeout: float = 120.0):
    """Embed texts via the local Ollama /api/embed endpoint. Returns an
    (n, d) float32 numpy array. Raises on transport errors (callers that must
    not fail use `resolve_uncertain`, which converts errors into fallbacks)."""
    import numpy as np
    out = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        req = urllib.request.Request(
            EMBED_URL,
            data=json.dumps({"model": model, "input": chunk}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            embs = json.loads(r.read())["embeddings"]
        out.extend(embs)
        if len(texts) > 200 and (s // batch) % 20 == 0:
            log.info("  embedded %d/%d", min(s + batch, len(texts)), len(texts))
    return np.asarray(out, dtype=np.float32)


def embed_available(model: str) -> bool:
    """Cheap probe: can Ollama serve one embedding for this model right now?"""
    try:
        embed_texts(["probe"], model, batch=1, timeout=10.0)
        return True
    except Exception as exc:
        log.warning("Embedding endpoint unavailable (%s): %s", model, repr(exc))
        return False


def load_head(path: str) -> Optional[Dict]:
    """Load a trained head artifact {pipeline, threshold, model, dataset,
    sklearn_version}. Returns None (with a warning) if missing or unreadable —
    the caller must fall back, never fabricate."""
    import os
    if not path or not os.path.exists(path):
        log.warning("Uncertain-head artifact not found: %s (run "
                    "train_uncertain_head.py). Uncertain blocks stay UNCERTAIN.", path)
        return None
    try:
        import joblib
        import sklearn
        head = joblib.load(path)
        assert "pipeline" in head and "threshold" in head and "model" in head
        saved_ver = head.get("sklearn_version")
        if saved_ver and saved_ver != sklearn.__version__:
            log.warning("Head %s was trained with scikit-learn %s (running %s) — "
                        "verify with tests/test_embedding_head.py.",
                        path, saved_ver, sklearn.__version__)
        return head
    except Exception as exc:
        log.warning("Failed to load uncertain-head %s: %s. Uncertain blocks stay "
                    "UNCERTAIN.", path, repr(exc))
        return None


def resolve_uncertain(blocks, encoder, event_map, head) -> List[Dict]:
    """Final NORMAL/SUSPICIOUS decision for a list of uncertain-band blocks.

    Returns one dict per block:
      {"verdict": "NORMAL"|"SUSPICIOUS", "proba": float, "fallback": None}
      or {"verdict": None, "proba": None, "fallback": "<reason>"}.
    Any failure mode yields an explicit fallback — the block keeps its
    UNCERTAIN routing label downstream. No decision is ever fabricated."""
    n = len(blocks)
    if head is None:
        return [{"verdict": None, "proba": None,
                 "fallback": FALLBACK_HEAD_MISSING}] * n
    model = head.get("model", "llama3.2:1b")
    if not embed_available(model):
        return [{"verdict": None, "proba": None,
                 "fallback": FALLBACK_OLLAMA_DOWN}] * n

    results: List[Dict] = []
    texts, idx = [], []
    for i, b in enumerate(blocks):
        try:
            t = block_text(b, encoder, event_map)
        except Exception as exc:
            log.warning("block_text failed for uncertain block %d: %s", i, repr(exc))
            t = ""
        if t:
            texts.append(t); idx.append(i)
            results.append(None)          # filled below
        else:
            results.append({"verdict": None, "proba": None,
                            "fallback": FALLBACK_INVALID_INPUT})
    if texts:
        try:
            X = embed_texts(texts, model)
            proba = head["pipeline"].predict_proba(X)[:, 1]
            thr = float(head["threshold"])
            for j, i in enumerate(idx):
                p = float(proba[j])
                results[i] = {"verdict": "SUSPICIOUS" if p >= thr else "NORMAL",
                              "proba": p, "fallback": None}
        except Exception as exc:
            log.warning("Embedding/head inference failed (%s) — all %d uncertain "
                        "blocks stay UNCERTAIN.", repr(exc), len(idx))
            for i in idx:
                results[i] = {"verdict": None, "proba": None,
                              "fallback": FALLBACK_EMBED_ERROR}
    return results

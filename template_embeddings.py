# =============================================================================
# preprocessing/template_embeddings.py — Semantic init for the event embedding.
#
# Today the model sees events as opaque integer IDs (E1, E2, ...), so it has no
# idea that E5 "Receiving block" and E9 "Received block" are related. This module
# turns each event's TEMPLATE TEXT into a vector, so semantically similar events
# start CLOSE TOGETHER in embedding space. That helps rare/ambiguous events and is
# the key lever for generalising to new log sources (an unseen template lands near
# similar known ones instead of being a fresh unknown ID).
#
# Method (no new dependencies — sklearn only, already installed):
#   1. TF-IDF over the 29 template strings  → a sparse (vocab, n_terms) matrix.
#   2. TruncatedSVD                          → dense (vocab, k) semantic vectors
#      (k = min(emb_dim, n_events-1); 29 templates span at most ~28 dims).
#   3. Place into an (vocab, emb_dim) matrix, fill any spare dims with small noise,
#      and scale to the model's usual init range (~U[-0.1, 0.1]).
#
# This is intentionally simple and explainable. A heavier sentence-transformer
# could be swapped in here, but TF-IDF already clusters "Receiving"/"Received".
# =============================================================================

import logging

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

log = logging.getLogger(__name__)


def build_template_embedding_matrix(encoder, embedding_dim, templates,
                                    seed: int = 42) -> torch.FloatTensor:
    """Return a (vocab_size, embedding_dim) tensor of semantic event vectors.

    Rows are aligned to `encoder.classes_` order (so row i = the embedding for the
    EventId the model uses as index i). Events with no template fall back to a small
    random row, so the function is safe even with a partial template map.

    Args:
        encoder       : fitted StreamingLabelEncoder (gives the EventId → index order).
        embedding_dim : target embedding width (must match the model's emb dim).
        templates     : dict EventId → template text (from load_event_templates()).
        seed          : RNG seed for the spare-dimension noise (reproducible).
    """
    rng = np.random.default_rng(seed)
    classes = encoder.classes_
    vocab = len(classes)

    # Template text per event, in the encoder's index order. Missing → empty string.
    docs = [templates.get(eid, "") or "" for eid in classes]
    n_have = sum(1 for d in docs if d.strip())
    log.info("Semantic embeddings: %d/%d events have template text.", n_have, vocab)

    # 1. TF-IDF (word + char n-grams help with 'Receiving' vs 'Received').
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z]+",
                          ngram_range=(1, 2), min_df=1)
    try:
        tfidf = vec.fit_transform(docs).toarray()
    except ValueError:
        # No usable vocabulary at all → fall back to pure random init.
        log.warning("No template vocabulary; using random embedding init.")
        return torch.empty(vocab, embedding_dim).uniform_(-0.1, 0.1)

    # 2. SVD → dense semantic vectors (capped at the achievable rank).
    k = max(1, min(embedding_dim, tfidf.shape[1] - 1, vocab - 1))
    # 'arpack' is stable/deterministic for these small (≤29-row) matrices; the default
    # randomized solver emits NaN-matmul warnings on near-degenerate inputs.
    svd = TruncatedSVD(n_components=k, algorithm="arpack", random_state=seed)
    sem = np.nan_to_num(svd.fit_transform(tfidf))        # (vocab, k); guard tiny/degenerate SVD

    # Normalise each event vector, then scale to the model's init range.
    norms = np.linalg.norm(sem, axis=1, keepdims=True)
    sem = sem / np.clip(norms, 1e-8, None)

    # 3. Assemble the (vocab, emb_dim) matrix: semantic dims first, small noise after.
    mat = rng.uniform(-0.02, 0.02, size=(vocab, embedding_dim)).astype(np.float32)
    mat[:, :k] = sem.astype(np.float32)
    # Scale the whole thing to roughly U[-0.1, 0.1] like the default init.
    mat = 0.1 * mat / max(np.abs(mat).max(), 1e-8)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)   # never inject NaN into the model
    return torch.from_numpy(mat)

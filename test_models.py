# =============================================================================
# tests/test_models.py — Regression suite for the model variants + attention.
#
# Plain-assert tests, no pytest dependency. Run from the repo root:
#     venv/bin/python tests/test_models.py
#
# Covers: the three variants build and forward to the right shapes; attention
# weights are valid (shape (B,T), sum to 1); the default model stays byte-compatible
# with the original architecture (so the saved checkpoint loads); composite_score
# is bounded; and the attention table + report section behave.
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                            # noqa: E402
import torch.nn as nn                                    # noqa: E402

from config import variant_flags                        # noqa: E402
from models.gru_model import GRUAnomalyDetector         # noqa: E402
from evaluation.evaluate import composite_score         # noqa: E402
from evaluation.attention import attention_table        # noqa: E402
from llm.explanation import _assemble_evidence_report   # noqa: E402
from training.train import FocalLoss                     # noqa: E402
from preprocessing.template_embeddings import build_template_embedding_matrix  # noqa: E402

_V, _B, _T = 29, 4, 20


def _make(variant):
    f = variant_flags(variant)
    return GRUAnomalyDetector(_V, 64, 128, 2, 0.3,
                              bidirectional=f["bidirectional"],
                              use_attention=f["use_attention"])


def test_all_variants_forward_shapes():
    """Each variant returns logits (B, vocab)."""
    x = torch.randint(0, _V, (_B, _T))
    for variant in ("gru", "bigru", "bigru_attention"):
        logits = _make(variant).eval()(x)
        assert tuple(logits.shape) == (_B, _V), (
            f"{variant}: logits shape {tuple(logits.shape)} != {(_B, _V)}")


def test_attention_weights_valid():
    """bigru_attention returns weights (B, T) that sum to 1; others return None."""
    x = torch.randint(0, _V, (_B, _T))
    _, w = _make("bigru_attention").eval()(x, return_attention=True)
    assert tuple(w.shape) == (_B, _T), f"attn shape {tuple(w.shape)} != {(_B, _T)}"
    assert torch.allclose(w.sum(dim=1), torch.ones(_B), atol=1e-5), "weights must sum to 1"
    assert (w >= 0).all(), "softmax weights must be non-negative"

    _, w_none = _make("bigru").eval()(x, return_attention=True)
    assert w_none is None, "non-attention variant must return None weights"


def test_default_is_backward_compatible():
    """Default constructor == the original 'gru' architecture (same state_dict keys),
    and the saved checkpoint (if present) loads with strict=True."""
    default_keys = set(GRUAnomalyDetector(_V, 64, 128, 2, 0.3).state_dict().keys())
    gru_keys     = set(_make("gru").state_dict().keys())
    assert default_keys == gru_keys, "default model must match the 'gru' variant"
    # The original architecture has no attention parameters.
    assert not any("attention" in k for k in default_keys), "default model must have no attention params"

    ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".checkpoints", "best_model.pt")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        GRUAnomalyDetector(_V, 64, 128, 2, 0.3).load_state_dict(sd, strict=True)


def test_composite_score_bounded():
    """composite_score stays within [0, 1] for arbitrary logits."""
    logits = torch.randn(_B, _V)
    y      = torch.randint(0, _V, (_B,))
    s = composite_score(logits, y, {"topk_k": 3, "entropy_weight": 0.35, "topk_miss_weight": 0.10})
    assert tuple(s.shape) == (_B,), f"score shape {tuple(s.shape)} != {(_B,)}"
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0, "scores must be in [0, 1]"


def test_attention_table_aggregates_and_sorts():
    """Repeated EventIds sum their weights; the table is sorted descending."""
    table = attention_table(["E5", "E22", "E5", "E11"], [0.1, 0.3, 0.4, 0.2])
    as_dict = dict(table)
    assert abs(as_dict["E5"] - 0.5) < 1e-9, "repeated E5 weights must sum (0.1+0.4)"
    weights = [w for _, w in table]
    assert weights == sorted(weights, reverse=True), "table must be sorted high→low"
    assert table[0][0] == "E5", "highest-weight event should sort first"


def test_report_attention_section_is_optional():
    """The 'Model Attention Focus' section appears only when attention is supplied."""
    base = dict(sequence=["E5", "E22"], predicted_event="E5", actual_event="E11",
                confidence=0.3, anomaly_score=0.7, classification="ANOMALY",
                prediction_correct=False,
                involved={"E5": "Receiving block", "E11": "Served block"},
                interpretation="E11 was observed.")
    assert "Model Attention Focus" not in _assemble_evidence_report(**base)
    with_attn = _assemble_evidence_report(**base, attention=[("E11", 0.42), ("E5", 0.31)])
    assert "Model Attention Focus" in with_attn
    assert "E11" in with_attn and "0.4200" in with_attn


class _Enc:
    """Minimal encoder stub (only .classes_ is needed by the embedding builder)."""
    def __init__(self, classes):
        self.classes_ = classes


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def test_focal_loss_reduces_to_ce_at_gamma_zero():
    """FocalLoss(gamma=0, no weights) == unweighted CrossEntropyLoss (the (1-p)^0 factor is 1)."""
    logits = torch.randn(_B, _V)
    y      = torch.randint(0, _V, (_B,))
    focal  = FocalLoss(gamma=0.0)(logits, y)
    ce     = nn.CrossEntropyLoss()(logits, y)
    assert torch.allclose(focal, ce, atol=1e-5), f"focal(γ=0)={focal} vs CE={ce}"


def test_focal_loss_downweights_easy_examples():
    """For a confident-correct example, focal loss (γ=2) is far below CE."""
    logits = torch.tensor([[10.0] + [0.0] * (_V - 1)])   # very confident on class 0
    y      = torch.tensor([0])
    focal  = float(FocalLoss(gamma=2.0)(logits, y))
    ce     = float(nn.CrossEntropyLoss()(logits, y))
    assert focal < ce, f"focal {focal} should be < CE {ce} on an easy example"


def test_semantic_init_keeps_state_dict_and_loads_matrix():
    """Supplying pretrained_embeddings must NOT change state_dict keys (so existing
    checkpoints still load) and must actually copy the matrix into the embedding."""
    emb = torch.randn(_V, 64) * 0.05
    m   = GRUAnomalyDetector(_V, 64, 128, 2, 0.3, pretrained_embeddings=emb)
    base_keys = set(GRUAnomalyDetector(_V, 64, 128, 2, 0.3).state_dict().keys())
    assert set(m.state_dict().keys()) == base_keys, "semantic init must not add/rename params"
    assert torch.allclose(m.embedding.weight.detach(), emb, atol=1e-5), "matrix not copied"


def test_template_embedding_matrix_clusters_similar_text():
    """build_template_embedding_matrix returns the right shape and places events with
    similar template text closer than dissimilar ones."""
    enc = _Enc(["E1", "E2", "E3"])
    templates = {
        "E1": "deleting block file path",
        "E2": "deleting block file path now",   # nearly identical to E1
        "E3": "receiving packet over the network",
    }
    mat = build_template_embedding_matrix(enc, embedding_dim=8, templates=templates)
    assert tuple(mat.shape) == (3, 8), f"shape {tuple(mat.shape)} != (3, 8)"
    assert _cos(mat[0], mat[1]) > _cos(mat[0], mat[2]), (
        "near-identical templates (E1,E2) should embed closer than dissimilar (E1,E3)")


_ALL_TESTS = [
    test_all_variants_forward_shapes,
    test_attention_weights_valid,
    test_default_is_backward_compatible,
    test_composite_score_bounded,
    test_attention_table_aggregates_and_sorts,
    test_report_attention_section_is_optional,
    test_focal_loss_reduces_to_ce_at_gamma_zero,
    test_focal_loss_downweights_easy_examples,
    test_semantic_init_keeps_state_dict_and_loads_matrix,
    test_template_embedding_matrix_clusters_similar_text,
]


def run_model_tests() -> None:
    for t in _ALL_TESTS:
        t()


if __name__ == "__main__":
    passed = 0
    for t in _ALL_TESTS:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\nAll {passed} model/variant tests passed.")

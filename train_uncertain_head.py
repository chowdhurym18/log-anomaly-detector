# =============================================================================
# train_uncertain_head.py — (re)builds the PRODUCTION uncertain-band head
# artifacts deterministically from the cached embedding NPZs.
#
# The head configurations are FROZEN to the dev-selected values recorded in
# outputs/uncertain_sets/{phase4_probe_llama3.2_1b,hdfs_uncertain_eval}.json —
# nothing here re-selects a model or a threshold:
#   BGL : LogisticRegression(C=0.1,  class_weight="balanced") · threshold 0.598
#   HDFS: LogisticRegression(C=0.01, class_weight="balanced") · threshold 0.76
#
# Before saving, the script HARD-ASSERTS that the refit head reproduces the
# published held-out confusion matrices exactly (BGL TP/FP/TN/FN 37/4/108/8;
# HDFS 328/2/967/0). This is a reproduction check of the single, already-spent
# test evaluation — it aborts on drift; it never tunes anything.
#
# Out: outputs/uncertain_sets/uncertain_head_{bgl,hdfs}.joblib
# Run: venv/bin/python train_uncertain_head.py [--datasets bgl,hdfs]
# =============================================================================

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.logging_utils import setup_logging

log = logging.getLogger(__name__)
SETS = Path("outputs/uncertain_sets")

# FROZEN dev-selected configs + the published held-out confusions they must
# reproduce. Do not edit without a new, pre-registered selection round.
FROZEN = {
    "bgl": {
        "npz": SETS / "phase4_embeddings_llama3.2_1b.npz",
        "regen": "venv/bin/python phase4_embedding_probe.py --model llama3.2:1b",
        "C": 0.1, "threshold": 0.598, "model": "llama3.2:1b",
        "expected_confusion": {"tp": 37, "fp": 4, "tn": 108, "fn": 8},
    },
    "hdfs": {
        "npz": SETS / "hdfs_embeddings_llama3.2_1b.npz",
        "regen": "venv/bin/python hdfs_uncertain_eval.py --model llama3.2:1b",
        "C": 0.01, "threshold": 0.76, "model": "llama3.2:1b",
        "expected_confusion": {"tp": 328, "fp": 2, "tn": 967, "fn": 0},
    },
}


def build_head(dataset: str) -> Path:
    spec = FROZEN[dataset]
    if not spec["npz"].exists():
        raise SystemExit(
            f"[{dataset}] embedding cache missing: {spec['npz']}\n"
            f"Regenerate it (needs Ollama + the frozen GRU):\n  {spec['regen']}")
    z = np.load(spec["npz"])
    Xtr, ytr = z["train_X"], z["train_y"]
    Xte, yte = z["test_X"], z["test_y"]
    log.info("[%s] train %s · test %s · C=%s · thr=%s", dataset,
             Xtr.shape, Xte.shape, spec["C"], spec["threshold"])

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=spec["C"], class_weight="balanced", max_iter=3000))
    pipeline.fit(Xtr, ytr)

    # Reproduction gate: the refit head must reproduce the PUBLISHED held-out
    # confusion exactly (this evaluation was already spent; abort on drift).
    pred = (pipeline.predict_proba(Xte)[:, 1] >= spec["threshold"]).astype(int)
    got = {"tp": int(((pred == 1) & (yte == 1)).sum()),
           "fp": int(((pred == 1) & (yte == 0)).sum()),
           "tn": int(((pred == 0) & (yte == 0)).sum()),
           "fn": int(((pred == 0) & (yte == 1)).sum())}
    exp = spec["expected_confusion"]
    assert got == exp, (
        f"[{dataset}] REPRODUCTION FAILED: refit head gives {got}, published {exp}. "
        "Do not ship — investigate sklearn/version or NPZ drift.")
    acc = (got["tp"] + got["tn"]) / len(yte)
    log.info("[%s] reproduction gate PASSED: %s (acc %.3f)", dataset, got, acc)

    out = SETS / f"uncertain_head_{dataset}.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": float(spec["threshold"]),
                 "model": spec["model"], "dataset": dataset,
                 "C": spec["C"], "sklearn_version": sklearn.__version__,
                 "n_train": int(len(ytr)), "dim": int(Xtr.shape[1]),
                 "test_confusion": got}, out)
    log.info("[%s] saved %s", dataset, out)
    return out


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Rebuild frozen uncertain-band head artifacts.")
    ap.add_argument("--datasets", default="bgl,hdfs")
    args = ap.parse_args()
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if ds not in FROZEN:
            raise SystemExit(f"Unknown dataset {ds!r} (know: {sorted(FROZEN)})")
        build_head(ds)
    log.info("Done. ✓")


if __name__ == "__main__":
    main()

# Focal Loss vs Weighted Cross-Entropy

Both trained on the same ~1M-row subset/split. Focal loss aims to lift the rare, zero-F1 classes (E7/E27/E28) and thus macro-F1.

| config | Top-1 | Macro-F1 | Weighted-F1 | Detection-F1 | Detection-AUROC | Detection-PR-AUC |
|---|---|---|---|---|---|---|
| weighted_ce | 91.13% | 0.4991 | 0.9117 | 0.1359 | 0.4837 | 0.0821 |
| focal | 89.15% | 0.4184 | 0.8931 | 0.1774 | 0.4123 | 0.0689 |

**Delta (focal − weighted_ce):** macro-F1 -0.0808, Top-1 -1.98pp, detection-F1 +0.0415.

_Verdict: focal no meaningful macro-F1 gain on this ~1M-row split._


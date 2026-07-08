# Semantic Embedding Init vs Baseline

Both trained on the same ~1M-row subset/split. The semantic model initialises the event embedding from template TEXT (TF-IDF+SVD) so similar events start close.

| config | Top-1 | Macro-F1 | Weighted-F1 | Detection-F1 | Detection-AUROC | Detection-PR-AUC |
|---|---|---|---|---|---|---|
| weighted_ce | 91.13% | 0.4991 | 0.9117 | 0.1359 | 0.4837 | 0.0821 |
| semantic | 91.32% | 0.4805 | 0.9131 | 0.1179 | 0.4827 | 0.0687 |

**Delta (semantic − weighted_ce):** macro-F1 -0.0187, Top-1 +0.19pp, detection-F1 -0.0180.

_Verdict: semantic no meaningful macro-F1 gain on this ~1M-row split._


# Anomaly-Scoring Comparison (Tasks 3 & 4)

_Model held fixed (**bgl_normal_only**); only the scoring rule varies. Honest held-out TEST block split; the F1 threshold is chosen on VALIDATION and applied to TEST. Best score chosen by validation PR-AUC._

- Test blocks: **9,427** · anomaly base rate **0.102** · val blocks **4,714**

**Best separator (by val PR-AUC): `combined_z(nll_max)+z(miss_count)`**

| Score (signal + aggregation) | AUROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| combined_z(nll_max)+z(miss_count) ⭐ | 0.9985 | **0.9946** | 0.9839 | 0.9523 | 0.9679 |
| nll_rolling_max(w=5) | 0.9973 | **0.9852** | 0.9740 | 0.9306 | 0.9518 |
| topk_miss_count | 0.9932 | **0.9595** | 0.9525 | 0.8725 | 0.9108 |
| topk_miss_frac | 0.9932 | **0.9595** | 0.9525 | 0.8725 | 0.9108 |
| nll_-logp_mean | 0.9863 | **0.9497** | 0.9822 | 0.8601 | 0.9171 |
| surprisal_1mp_mean | 0.9710 | **0.8932** | 0.9654 | 0.7513 | 0.8450 |
| msp_self_uncertainty (1-MSP, max) | 0.9703 | **0.8652** | 0.9446 | 0.8311 | 0.8842 |
| surprisal_1mp_max | 0.9898 | **0.9079** | 0.7932 | 0.9979 | 0.8839 |
| nll_-logp_max | 0.9897 | **0.9077** | 0.7932 | 0.9979 | 0.8839 |
| nll_perevent_z_max | 0.8572 | **0.2976** | 0.3560 | 0.7544 | 0.4837 |
| nll_perevent_z_mean | 0.6526 | **0.2310** | 0.2801 | 0.7254 | 0.4042 |

## Reading this

- **Signal:** `-log p` (NLL) is the proper surprise — its unbounded tail separates a p≈1e-6 transition from a merely-unlikely one, unlike `1 - p` which saturates at 1. **MSP** is the model's confidence in its OWN top event (the hybrid-pipeline signal); `1 - MSP` is its self-uncertainty.
- **Aggregation (Task 4):** `max` flags the single most surprising transition (DeepLog's rule); `count` is the number of out-of-top-k violations; `mean` is the average surprise; `rolling_max` is the most surprising sustained BURST.
- **Combined** standardises and sums the two strongest orthogonal signals (peak surprise + violation count), z-fit on validation.


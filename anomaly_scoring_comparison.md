# Anomaly-Scoring Comparison (Tasks 3 & 4)

_Model held fixed (**normal_only**); only the scoring rule varies. Honest held-out TEST block split; the F1 threshold is chosen on VALIDATION and applied to TEST. Best score chosen by validation PR-AUC._

- Test blocks: **115,013** · anomaly base rate **0.029** · val blocks **57,506**

**Best separator (by val PR-AUC): `nll_-logp_mean`**

| Score (signal + aggregation) | AUROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| nll_-logp_mean ⭐ | 0.8917 | **0.6917** | 0.8018 | 0.5814 | 0.6740 |
| combined_z(nll_max)+z(miss_count) | 0.8358 | **0.6728** | 0.7212 | 0.7150 | 0.7181 |
| surprisal_1mp_max | 0.8348 | **0.6539** | 0.7935 | 0.6606 | 0.7210 |
| nll_-logp_max | 0.8348 | **0.6536** | 0.7935 | 0.6606 | 0.7210 |
| nll_rolling_max(w=5) | 0.8756 | **0.6515** | 0.7697 | 0.5787 | 0.6607 |
| topk_miss_frac | 0.7704 | **0.4837** | 0.8000 | 0.4834 | 0.6026 |
| surprisal_1mp_mean | 0.8210 | **0.4723** | 0.7226 | 0.3759 | 0.4945 |
| topk_miss_count | 0.7622 | **0.2983** | 0.7013 | 0.2865 | 0.4068 |
| nll_perevent_z_mean | 0.6977 | **0.2241** | 0.3778 | 0.3524 | 0.3647 |
| nll_perevent_z_max | 0.5238 | **0.1529** | 0.6467 | 0.1957 | 0.3004 |
| msp_self_uncertainty (1-MSP, max) | 0.4372 | **0.0341** | 0.0683 | 0.0425 | 0.0524 |

## Reading this

- **Signal:** `-log p` (NLL) is the proper surprise — its unbounded tail separates a p≈1e-6 transition from a merely-unlikely one, unlike `1 - p` which saturates at 1. **MSP** is the model's confidence in its OWN top event (the hybrid-pipeline signal); `1 - MSP` is its self-uncertainty.
- **Aggregation (Task 4):** `max` flags the single most surprising transition (DeepLog's rule); `count` is the number of out-of-top-k violations; `mean` is the average surprise; `rolling_max` is the most surprising sustained BURST.
- **Combined** standardises and sums the two strongest orthogonal signals (peak surprise + violation count), z-fit on validation.


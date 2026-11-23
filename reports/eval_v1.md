# Evaluation v1: ibm_hi_small

Synthetic data (IBM AMLworld). Account level. Positives = **new** mules in the period; known mules, sources and sinks excluded. Mean ± std over seeds (LightGBM and TXN_ONLY: 5 seeds; ablations: 3; others: 1).

## Test (65 new mules among 349,637 accounts; chance PR-AUC 0.0002)

| Approach | PR-AUC | ROC-AUC | P@50 | P@100 | R@100 | R@500 | FPR top 1% |
|---|---|---|---|---|---|---|---|
| RULES | 0.0010 ± 0.0000 | 0.670 | 0.020 | 0.020 | 0.031 | 0.046 | 0.0100 |
| TXN_ONLY | 0.0618 ± 0.0072 | 0.935 | 0.128 | 0.092 | 0.142 | 0.225 | 0.0099 |
| LIGHTGBM | 0.1806 ± 0.0152 | 0.988 | 0.284 | 0.174 | 0.268 | 0.431 | 0.0099 |
| LOGREG | 0.0018 ± 0.0000 | 0.926 | 0.000 | 0.000 | 0.000 | 0.015 | 0.0100 |
| RANDOM_FOREST | 0.0663 ± 0.0000 | 0.970 | 0.200 | 0.130 | 0.200 | 0.385 | 0.0099 |
| ISOLATION_FOREST | 0.0007 ± 0.0000 | 0.785 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0100 |
| FULL-STRUCTURAL | 0.2578 ± 0.0178 | 0.989 | 0.327 | 0.183 | 0.282 | 0.528 | 0.0099 |
| FULL-TEMPORAL | 0.0403 ± 0.0043 | 0.953 | 0.147 | 0.080 | 0.123 | 0.190 | 0.0099 |
| FULL-FLOW | 0.0709 ± 0.0054 | 0.983 | 0.200 | 0.133 | 0.205 | 0.369 | 0.0099 |
| FULL-RULES | 0.1833 ± 0.0110 | 0.987 | 0.273 | 0.167 | 0.256 | 0.436 | 0.0099 |
| FULL-RELATIVE | 0.1674 ± 0.0106 | 0.988 | 0.293 | 0.173 | 0.267 | 0.426 | 0.0099 |

## Validation (210 new mules among 231,535 accounts; chance PR-AUC 0.0009)

| Approach | PR-AUC | ROC-AUC | P@50 | P@100 | R@100 | R@500 | FPR top 1% |
|---|---|---|---|---|---|---|---|
| RULES | 0.0012 ± 0.0000 | 0.555 | 0.000 | 0.010 | 0.005 | 0.010 | 0.0100 |
| TXN_ONLY | 0.1707 ± 0.0055 | 0.920 | 0.528 | 0.394 | 0.188 | 0.293 | 0.0096 |
| LIGHTGBM | 0.2859 ± 0.0048 | 0.972 | 0.704 | 0.510 | 0.243 | 0.410 | 0.0094 |
| LOGREG | 0.0060 ± 0.0000 | 0.874 | 0.000 | 0.000 | 0.000 | 0.019 | 0.0099 |
| RANDOM_FOREST | 0.2182 ± 0.0000 | 0.965 | 0.620 | 0.410 | 0.195 | 0.357 | 0.0095 |
| ISOLATION_FOREST | 0.0020 ± 0.0000 | 0.640 | 0.000 | 0.000 | 0.000 | 0.019 | 0.0100 |
| FULL-STRUCTURAL | 0.2777 ± 0.0014 | 0.971 | 0.693 | 0.510 | 0.243 | 0.392 | 0.0094 |
| FULL-TEMPORAL | 0.1262 ± 0.0010 | 0.909 | 0.427 | 0.247 | 0.117 | 0.225 | 0.0097 |
| FULL-FLOW | 0.2214 ± 0.0104 | 0.970 | 0.573 | 0.420 | 0.200 | 0.344 | 0.0095 |
| FULL-RULES | 0.2882 ± 0.0055 | 0.973 | 0.693 | 0.517 | 0.246 | 0.406 | 0.0094 |
| FULL-RELATIVE | 0.1974 ± 0.0039 | 0.965 | 0.547 | 0.357 | 0.170 | 0.346 | 0.0095 |

## Top features (LightGBM gain)

- max_in_pctl: 13245.2
- in_degree: 8853.2
- new_counterparties_24h_ratio: 6562.1
- nbr_mean_degree_log: 6123.8
- self_txn_share: 6041.7
- age_hours: 5876.2
- pagerank: 4341.4
- holding_p10_min: 4116.1
- channel_entropy: 4064.1
- betweenness_local: 3629.8
- net_flow_ratio: 3494.8
- night_share: 3335.3
- max_out_vs_own_median: 3255.8
- mean_amount_pctl: 2859.6
- max_in_vs_own_median: 2843.6

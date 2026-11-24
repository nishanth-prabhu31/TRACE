# UPI-style stress test (ibm_hi_small, model v2)

Stress test only: injected rings were never trained on. Synthetic background (IBM), synthetic rings.

| Variant | Detector | Mules caught | Recall | Real accounts also flagged |
|---|---|---|---|---|
| FRESH | model_likely | 0 / 122 | 0.00 | 118 |
| FRESH | model_watch | 0 / 122 | 0.00 | 1,126 |
| FRESH | rules_any | 122 / 122 | 1.00 | 489 |
| FRESH | either | 122 / 122 | 1.00 | 1,596 |
| AGED | model_likely | 0 / 124 | 0.00 | 122 |
| AGED | model_watch | 16 / 124 | 0.13 | 1,137 |
| AGED | rules_any | 124 / 124 | 1.00 | 490 |
| AGED | either | 124 / 124 | 1.00 | 1,610 |

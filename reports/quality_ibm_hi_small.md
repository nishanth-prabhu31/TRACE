# Quality report: ibm_hi_small

**Overall: ✅ PASS**

| Check | Status | Value | Detail |
|---|---|---|---|
| C1 Format | ✅ PASS | 2e-06 | 9 of 5,078,345 rows dropped as broken (0.0002%); 1,108 tail rows removed on purpose |
| C2 Labels | ✅ PASS | 795 | NORMAL 509,216, SINK 2,826, SOURCE 2,241, MULE 795 |
| C3 Time spread | ✅ PASS | 65 | new mules: train 520, validation 210, test 65; known mules still active in test 602 (new share 0.097) |
| C4 Giveaway test | ✅ PASS | 0.0064 | highest single-column PR-AUC 0.006 (txn:channel); chance level is about 0.0011 for transactions and 0.0016 for accounts. All: txn:amount 0.004, txn:channel 0.006, txn:currency_original 0.002, txn:hour 0.001, txn:sender_bank 0.005, txn:receiver_bank 0.002, account:main_channel 0.004, account:main_currency 0.002, account:mean_amount 0.002, account:bank 0.002 |
| C5 Overlap | ✅ PASS | 0.0176 | best one-threshold F1 0.018 (out_degree). All: in_degree 0.016, out_degree 0.018, total_amount 0.010 |
| C6 Graph sanity | ✅ PASS | 0.9987 | 99.9% of mules are in the largest connected part (371,915 of 422,656 accounts); 93.0% of the top-1% most connected accounts are normal (normal hubs exist) |
| C7 Compatibility | ✅ PASS | 0 | 41 features in 20220911T0000.parquet; constant or empty: none |

Thresholds: config/settings.yaml `quality_gate`. Meaning of each check: docs/02-DATA.md section 8.

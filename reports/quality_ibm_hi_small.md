# Data quality report: ibm_hi_small

**Overall: ⚠️ WARN**

Label-based checks (C4, C5, C6) use only the train and validation periods.

| # | Check | Status | Value | Detail |
|---|---|---|---|---|
| C1 | Format | ✅ PASS | 0.0 | 0.0002% of rows dropped as broken (9 rows); 1,108 tail rows removed on purpose |
| C2 | Labels | ✅ PASS | 795.0 | 795 mule, 2,241 source, 2,826 sink, 509,216 normal |
| C3 | Time spread | ⚠️ WARN | 202.0 | mules: 340 train, 253 new in validation, 202 new in test; 60% of the 667 mules active in the test period were not mules in the train period |
| C4 | Giveaway test | ✅ PASS | 0.0078 | highest one-column PR-AUC: transaction:channel = 0.008 (fitted on train, scored on validation). All: transaction:channel 0.008, account:bank 0.007, transaction:sender_bank 0.006, transaction:amount 0.005, account:channel 0.003, account:amount 0.003, account:currency_original 0.003, account:hour 0.002, transaction:receiver_bank 0.002, transaction:currency_original 0.002, transaction:hour 0.001 |
| C5 | Overlap | ✅ PASS | 0.0171 | best one-threshold F1 (data up to end of validation): in_degree 0.014, out_degree 0.017, total_amount 0.008 |
| C6 | Graph sanity | ✅ PASS | 0.9983 | 100% of mules are in the largest connected part; 80 of the 100 best-connected accounts are normal |
| C7 | Compatibility | ✅ PASS | 0.0 | 41 features in 10 files; constant: none; all empty: none |

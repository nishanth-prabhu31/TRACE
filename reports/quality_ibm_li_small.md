# Data quality report: ibm_li_small

**Overall: ⚠️ WARN**

Label-based checks (C4, C5, C6) use only the train and validation periods.

| # | Check | Status | Value | Detail |
|---|---|---|---|---|
| C1 | Format | ✅ PASS | 0.0 | 0.0001% of rows dropped as broken (8 rows); 223 tail rows removed on purpose |
| C2 | Labels | ✅ PASS | 323.0 | 323 mule, 1,988 source, 2,883 sink, 700,711 normal |
| C3 | Time spread | ⚠️ WARN | 91.0 | mules: 141 train, 91 new in validation, 91 new in test; 58% of the 286 mules active in the test period were not mules in the train period |
| C4 | Giveaway test | ✅ PASS | 0.004 | highest one-column PR-AUC: account:bank = 0.004 (fitted on train, scored on validation). All: account:bank 0.004, transaction:channel 0.003, transaction:amount 0.001, account:currency_original 0.001, transaction:sender_bank 0.001, transaction:receiver_bank 0.001, account:amount 0.001, account:channel 0.001, transaction:currency_original 0.001, account:hour 0.001, transaction:hour 0.001 |
| C5 | Overlap | ✅ PASS | 0.0099 | best one-threshold F1 (data up to end of validation): in_degree 0.004, out_degree 0.010, total_amount 0.003 |
| C6 | Graph sanity | ✅ PASS | 0.9957 | 100% of mules are in the largest connected part; 80 of the 100 best-connected accounts are normal |
| C7 | Compatibility | ✅ PASS | 0.0 | 41 features in 10 files; constant: none; all empty: none |

# Replay: caught in time (ibm_hi_small, model v2)

Synthetic data. Steps every 1 h through the test period, so lead times are accurate to about 1 h. Rings: PATTERN_FILE (complete attempts only).

| Measure | Value |
|---|---|
| dataset | ibm_hi_small |
| model_version | v2 |
| step_hours | 1 |
| ring_definition | PATTERN_FILE (complete attempts only) |
| likely_mule_cut | 40.0 |
| rings | 73 |
| caught_before_cashout | 55 |
| caught_after_cashout | 5 |
| never_caught | 13 |
| share_before_cashout | 0.7534 |
| lead_minutes_median | 1062.0 |
| lead_minutes_p25_p75 | [633.5, 1906.0] |
| strict_caught_after_start_before_cashout | 37 |
| strict_share | 0.5068 |
| strict_lead_minutes_median | 1138.0 |
| alert_before_ring_start | 18 |

| Pattern | Rings | Caught before cash-out |
|---|---|---|
| BIPARTITE | 11 | 6 |
| CYCLE | 11 | 9 |
| FAN_IN | 6 | 4 |
| FAN_OUT | 9 | 5 |
| GATHER_SCATTER | 9 | 9 |
| RANDOM | 8 | 7 |
| SCATTER_GATHER | 6 | 6 |
| STACK | 13 | 9 |

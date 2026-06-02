# Native Real-Demand Transit Control Validation

native simulator passenger loop with public AFC/APC profile mapping, not exact AFC/APC OD geometry

## Sources

| source | rows | series | bins/hour | boundary |
|---|---:|---:|---:|---|
| afc | 1000 | 2 | 1 | AFC station entries, not onboard load or OD |
| apc | 1000 | 2 | 2 | APC route boardings, not onboard occupancy/alighting/OD |

## Paired Checks

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_real_demand_control_score | supported | control_score | 6 | +99.6725 | +62.3044 | +137.0299 | 1.00 |
| native_real_demand_ep_reward | supported | ep_reward | 6 | +98.7658 | +59.9997 | +137.1515 | 1.00 |
| native_real_demand_avg_wait_min | positive_mixed | avg_wait_min | 6 | -0.0830 | -0.2248 | +0.0567 | 0.67 |
| native_real_demand_native_avg_board_wait_min | positive_mixed | native_avg_board_wait_min | 6 | -0.0833 | -0.2251 | +0.0564 | 0.67 |
| native_real_demand_native_alighted_pax | not_supported | native_alighted_pax | 6 | -4.8333 | -9.8333 | -0.8333 | 0.17 |
| native_real_demand_native_avg_onboard_load | inconclusive | native_avg_onboard_load | 6 | -0.0001 | -0.0005 | +0.0003 | 0.33 |

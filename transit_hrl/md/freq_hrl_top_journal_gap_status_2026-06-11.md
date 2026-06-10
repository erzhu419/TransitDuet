# Freq-HRL Top-Journal Gap Status - 2026-06-11

This note records the current state after the latest native promotion, real-demand, leakage, and order-book replay updates. It is intentionally conservative: a path is marked supported only when the available paired validation supports the stated claim.

## Current Claim State

| Gap | Current status | Evidence | Remaining work |
| --- | --- | --- | --- |
| Native learned promotion reward/wait | Partially supported | `scheduler_native_promotion_risk_banded_delta_floor_v42_512seed_merged` supports reward, wait, and score under persistent stress. | Cross-stress replication is not closed. |
| v42 cross-stress replication | Not supported on OD shift | `transit_native_promotion_v42_odshift_512seed_merged` has nearly no active replanning: reward delta is -0.0060 with CI upper at 0, wait/score are only inconclusive. | The odshift regime needs a stress-specific promotion policy, not just the persistent-stress v42 gate. |
| v43 pressure override for OD shift | Smoke only, do not scale as-is | First shard in `scheduler_native_promotion_v43_odshift_pressure_smoke64_shard_0_0_11` raises wait-aware replan count but reward becomes negative and wait is worse on mean. | Retune the pressure override with reward-floor and value-guard selection before any 512-seed expansion. |
| Real-demand wait/alighting | Not strictly supported | `transit_native_real_demand_alighting_wait_v4_24pair_merged` supports control score and episode reward, but wait, alighted pax, and completed throughput deltas remain zero or worse. | Use high-pressure real-shape demand scaling and throughput-aware action constraints. |
| Native leakage no-tradeoff | Partial | `leakage_no_tradeoff_matrix_v4_indexed` now indexes native real-demand metrics including `LowerLFDrift`; v4 drift reduction is still inconclusive. | Need native runs where drift reduction and reward/wait noninferiority hold together. |
| Real L2/L3 order-book replay | Mechanism ready, data claim not closed | The manifest validator now separates explicit real or venue-grade L2/L3 feeds from fixture/synthetic/sample files. | Add actual venue-grade L2/L3 event data and run large replay with queue-priority matching. |

## Code Changes Made For This Stage

1. `top_journal_unified_matrix.py` now prefers the v42 native promotion artifact for C1 and requires explicit real or venue-grade L2/L3 coverage for the order-book claim.
2. `native_real_demand_control_validation.py` adds the guarded `alighting_wait_v4` control profile and `--demand-scale-multiplier` for high-pressure real-shape stress tests.
3. `merge_native_real_demand_shards.py` preserves demand-scale metadata in merged outputs.
4. `leakage_no_tradeoff_matrix.py` indexes the new native real-demand v4 artifact and reports native drift/performance tradeoffs more strictly.
5. `native_promotion_replan_validation.py` adds the `odshift_pressure_replay_v43` profile, which is useful as a negative smoke result but should not be scaled without retuning.
6. `order_book_large_replay_manifest_validation.py` now treats real-data coverage as a source-quality claim, not merely as CSV availability.

## Next Experiments

1. Real-demand high-pressure replay: run `alighting_safe_v2` or a tuned v5 with `--demand-scale-multiplier 1.3-1.5`, then require wait/alighting/throughput improvement without reward loss.
2. OD-shift promotion retune: keep v43's pressure trigger, but add stricter reward-floor candidate selection and reject shifts that do not change the timetable enough to affect wait.
3. Native leakage no-tradeoff: use adaptive drift penalty instead of hard pressure, and report joint pass/fail on `LowerLFDrift`, reward, wait, and throughput.
4. Order-book replay: build a manifest with explicit `source_type: real` or `source_type: venue_grade` for both L2 and L3, then rerun the large replay matrix.

## Bottom Line

The implementation is stronger than the previous state because the matrix now separates supported claims from stress-specific failures and source-quality placeholders. The top-journal gap is still not closed: native promotion needs cross-stress reward/wait support, real-demand needs strict wait/alighting improvement, native leakage needs a no-tradeoff run, and order-book replay still needs real venue-grade L2/L3 data.

# Freq-HRL Top-Journal Gap Status - 2026-06-11

This note records the current state after the latest native promotion, real-demand, leakage, and order-book replay updates. It is intentionally conservative: a path is marked supported only when the available paired validation supports the stated claim.

## Current Claim State

| Gap | Current status | Evidence | Remaining work |
| --- | --- | --- | --- |
| Native learned promotion reward/wait | Partially supported | `scheduler_native_promotion_risk_banded_delta_floor_v42_512seed_merged` supports reward, wait, and score under persistent stress. | Cross-stress replication is not closed. |
| v42 cross-stress replication | Not supported on OD shift | `transit_native_promotion_v42_odshift_512seed_merged` has nearly no active replanning: reward delta is -0.0060 with CI upper at 0, wait/score are only inconclusive. | The odshift regime needs a stress-specific promotion policy, not just the persistent-stress v42 gate. |
| v43 pressure override for OD shift | Smoke only, do not scale as-is | First shard in `scheduler_native_promotion_v43_odshift_pressure_smoke64_shard_0_0_11` raises wait-aware replan count but reward becomes negative and wait is worse on mean. | Retune the pressure override with reward-floor and value-guard selection before any 512-seed expansion. |
| v44 OD-shift reward-floor wait-aware replan | Noninferiority supported, strong improvement not closed | `scheduler_native_promotion_v44_odshift_reward_floor_smoke64_merged_retry3_retry4` has 64 seeds: reward delta +2.4216 with CI [0.0000, 6.8108], wait delta -0.00022 min with CI [-0.00059, 0.0000]. Reward/wait noninferiority is supported, but improvement status remains inconclusive. | Scale or stress-register only if the next profile increases active replans without relying mostly on value-guard rejection. |
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
7. `merge_native_promotion_shards.py` now reports only actually observed variants and no longer emits underpowered checks for promotion variants that were not run in a shard set.

## Next Experiments

1. Real-demand high-pressure replay: run `alighting_safe_v2` or a tuned v5 with `--demand-scale-multiplier 1.3-1.5`, then require wait/alighting/throughput improvement without reward loss.
2. OD-shift promotion retune: keep v43's pressure trigger, but add stricter reward-floor candidate selection and reject shifts that do not change the timetable enough to affect wait.
3. Native leakage no-tradeoff: use adaptive drift penalty instead of hard pressure, and report joint pass/fail on `LowerLFDrift`, reward, wait, and throughput.
4. Order-book replay: build a manifest with explicit `source_type: real` or `source_type: venue_grade` for both L2 and L3, then rerun the large replay matrix.

## Bottom Line

The implementation is stronger than the previous state because the matrix now separates supported claims from stress-specific failures and source-quality placeholders. The top-journal gap is still not closed: native promotion needs cross-stress reward/wait support, real-demand needs strict wait/alighting improvement, native leakage needs a no-tradeoff run, and order-book replay still needs real venue-grade L2/L3 data.

After the v44 scheduler rerun, OD-shift promotion has moved from broken smoke to valid no-harm evidence. It still does not establish the stronger global reward/wait improvement claim.

## 2026-06-11 Follow-Up Patch

This patch turns the remaining paper gaps into explicit runnable evidence paths:

1. Native promotion: added `odshift_reward_floor_active_v45`, a v44 successor that lowers the value guard enough to create active OD-shift replans while keeping reward-floor, throughput-floor, and adaptive-drift guards.
2. Native real demand: added `alighting_throughput_v5`, a high-pressure AFC/APC profile focused on wait, alighting, completed throughput, and bounded lower-HF rescue.
3. Leakage no-tradeoff: indexed the upcoming native real-demand v5 artifact in `leakage_no_tradeoff_matrix.py` so drift reduction and performance no-harm are judged in the same native domain.
4. Strong baselines/ablations: added `baseline_ablation_matrix.py`, which computes paired Freq-HRL deltas against vanilla, raw HRL, all-frequency, swapped, no-promotion, no-leakage, LF-only, and HF-only baselines across identical seeds/stress scenarios.
5. Cross-stress promotion: extended `top_journal_unified_matrix.py` with C7 so persistent-stress success and OD-shift no-harm/improvement are reported separately instead of folded into one global claim.
6. Order-book evidence: added `source_quality_status` to the L2/L3 manifest runner. Fixture/sample data can only be `mechanism_only`; supported source-quality requires real or venue-grade L2 and L3 feeds.
7. Theory: added Proposition 8, a sufficient no-tradeoff margin condition that prevents the paper from claiming leakage no-tradeoff without same-domain paired performance slack.

Current expected status after this patch:

- C1 can remain supported for the best native persistent-stress promotion artifact.
- C7 should remain partial until v45 or another OD-shift profile supports reward and wait improvement CIs.
- C2 should remain partial until v5 high-pressure real-demand validation closes wait/alighting/throughput.
- C5 should remain partial until native real-demand or trading leakage jointly supports drift reduction and reward/wait no-harm.
- C8 should become explicit rather than implicit once `baseline_ablation_matrix` is generated from the current pressure/performance artifacts.

## 2026-06-11 Follow-Up Patch 2

Merged scheduler evidence and tightened the remaining paper-claim boundaries:

1. OD-shift promotion v45 was merged at 512 seeds. It is valid no-harm evidence, but not a strong improvement result: reward delta is +5.3845 with CI [-3.1094, +19.4043], wait delta is +0.00292 min with CI [-0.00037, +0.00929]. Reward/wait noninferiority is supported; reward and wait improvement remain inconclusive/not supported.
2. Native real-demand v5 was merged at 48 seed indices across AFC/APC, producing 96 paired source-seed comparisons. Score and reward are supported, wait is nearly improved but still inconclusive, and alighting/throughput remain not supported.
3. Added `odshift_reward_wait_guard_v46`, a narrower OD-shift profile that keeps active wait-aware replanning but tightens pressure, candidate-scale, throughput-floor, and adaptive-drift accept regions after v45 showed a small wait regression.
4. Added `throughput_safe_wait_v6`, a native AFC/APC profile that lowers lower-HF rescue side effects and adds stricter throughput/fleet floors. This is the next profile to validate for strict wait/alighting/throughput improvement and native leakage no-tradeoff.
5. Added wait-proxy and completed-throughput noninferiority checks to native real-demand validation, so no-harm and strict-improvement claims are separated for `avg_wait_min`, `native_avg_board_wait_min`, `native_alighted_pax`, and `native_completed_throughput_pax`.
6. Tightened `leakage_no_tradeoff_matrix.py`: native real-demand no-tradeoff now requires same-domain drift plus core reward/score/wait/alighting/throughput evidence, with a separate `no_tradeoff_strict_supported` verdict.
7. Tightened order-book manifest source quality: `venue_grade_ready` now requires paired L2 and L3 sessions with venue, symbol, session metadata and venue-grade/price-time semantics. Real-but-unpaired files are reported as `real_unpaired_or_metadata_incomplete`.
8. Added C9 to `top_journal_unified_matrix.py` for the pre-registered pressure regimes: stationary low noise, stationary high noise, localized burst, persistent shift, and OOD period. Current pressure matrix supports all five, so C9 is supported.
9. Added Proposition 9 to the theory appendix: global stress-generalization is an intersection claim over the pre-registered regime set; missing or failed regimes must be reported as claim boundaries.

Current unified matrix after this patch:

- C1 supported: best native persistent-stress promotion remains supported.
- C2 not supported under the stricter claim: real-demand score/reward are supported, but strict wait/alighting/throughput improvement is not closed.
- C3 partial: L2/L3 replay path exists, but venue-grade paired L2/L3 sessions are still missing.
- C4 supported: advanced encoder evidence spans current synthetic/real-demand domains.
- C5 partial: surrogate leakage no-tradeoff is strict-supported, but native/trading no-tradeoff is not yet closed.
- C6 supported: theory appendix now has 9 theorem/proposition rows.
- C7 partial: persistent promotion is supported, OD-shift remains no-harm rather than strong reward/wait improvement.
- C8 partial: baseline/ablation table is broad, but `no_promotion` Sharpe remains inconclusive.
- C9 supported: five pre-registered pressure regimes are covered by the pressure matrix.

Next scheduler validations should target:

1. `odshift_reward_wait_guard_v46` at 512 seeds.
2. `throughput_safe_wait_v6` across at least the same 48 AFC/APC seed indices as v5.
3. A refreshed leakage matrix after v6, using the strict native real-demand core metrics.
4. A venue-grade L2/L3 manifest once real exchange data is available.

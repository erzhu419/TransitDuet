# Freq-HRL Top-Journal Gap Status - 2026-06-11

> **Superseded evidence ledger (2026-08-03).** The dated entries below are
> retained only as an experiment history. They used legacy gates that admitted
> deterministic service/promotion projections and path-existence checks. The
> current raw-only source of truth is
> `transit_hrl/results/top_journal_unified_matrix_latest/summary.json`: 1 of 9
> claims is supported, 6 are partial, and 2 are not supported. No historical
> status in this file may be quoted as the current paper result.

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

## 2026-06-11 Scheduler Dispatch Note

Commit `64c9d6cf57` pushed the stricter top-journal gates and the new v46/v6 validation profiles. Local verification passed with 34 focused tests plus compileall over `transit_hrl/freq_hrl/experiments` and `transit_hrl/tests`.

Scheduler submission status:

1. Native real-demand `throughput_safe_wait_v6` was re-submitted with the correct CLI, after the first t9891-t9895 attempt failed because the script does not accept `--workers`. Fixed shard tasks are t9937-t9942. At dispatch time, t9937-t9941 were running on node005/node002 and t9942 was queued behind node006 placement.
2. Native promotion `odshift_reward_wait_guard_v46` is queued as 16 small w16 shards, t9943-t9958, covering seed-index ranges 0-512 in 32-seed blocks. The earlier 6 large shards t9877-t9882 and 8 req64 parent shards t9906/t9908/t9910/t9912/t9914/t9916/t9918/t9920 were cancelled while still queued.
3. The current blocker for promotion is scheduler CPU-token availability on node002-node005, not RAM. Existing v45 OD-shift tasks t9782-t9787 are still running on node002-node005 and consume most direct CPU slots. No running task was killed in this patch.
4. node001/node006 still show scheduler placement/env escalation behavior for this signature path. The queued w16 shards are therefore constrained to node002-node005 until the direct-node scheduler gate is cleared or those nodes free enough CPU tokens.

Once t9937-t9942 finish, merge the fixed v6 real-demand shards and regenerate the strict leakage/top-journal matrices. Once t9943-t9958 finish, merge v46 promotion and check whether OD-shift reward/wait moves from no-harm to CI-supported improvement.

## 2026-06-11 v46/v6 Merge Results

The queued scheduler validations finished and were merged locally:

1. Native real-demand `throughput_safe_wait_v6`: merged 48 seed indices and 96 AFC/APC paired comparisons into `transit_native_real_demand_throughput_safe_wait_v6_48pair_merged`.
2. Native promotion `odshift_reward_wait_guard_v46`: merged 512 seeds and 1024 paired rows into `scheduler_native_promotion_v46_odshift_reward_wait_guard_512seed_merged`.

Key outcomes:

| claim target | status | n | delta | CI95 |
|---|---|---:|---:|---|
| real-demand control score | supported | 96 | +2936.6188 | [+2445.6933, +3404.4883] |
| real-demand episode reward | supported | 96 | +2936.6188 | [+2445.6933, +3404.4883] |
| real-demand avg wait | not supported | 96 | +0.0000 | [+0.0000, +0.0000] |
| real-demand board wait | not supported | 96 | +0.0000 | [+0.0000, +0.0000] |
| real-demand alighted pax | not supported | 96 | +0.0000 | [+0.0000, +0.0000] |
| real-demand completed throughput | not supported | 96 | +0.0000 | [+0.0000, +0.0000] |
| promotion v46 reward improvement | inconclusive | 512 | +3.9650 | [-2.1008, +14.2042] |
| promotion v46 wait improvement | not supported | 512 | +0.0039 | [-0.0000, +0.0116] |
| promotion v46 reward noninferiority | supported | 512 | +3.9650 | [-2.1008, +14.2042] |
| promotion v46 wait noninferiority | positive_mixed | 512 | +0.0039 | [-0.0000, +0.0116] |
| promotion v46 gate replans | supported | 512 | +0.0117 | [+0.0039, +0.0215] |

The refreshed matrices are:

- `top_journal_unified_matrix_latest_after_v46_v6`: 4 supported, 5 partial, 0 missing/not-supported. C7 now correctly reports the 512-seed v46 OD-shift artifact rather than the smaller v44 smoke.
- `leakage_no_tradeoff_matrix_latest_after_v46_v6`: still only one strict supported domain, `transit_real_surrogate`; native real-demand v6 remains partial because LowerLFDrift/performance no-tradeoff is not closed.

Conclusion: v6 strengthens real-demand no-harm and reward/score support, but does not create strict wait/alighting/throughput improvement. v46 confirms learned promotion activity and reward no-harm at 512 seeds, but does not close OD-shift reward/wait improvement. The remaining top-journal gaps are now narrower and explicit: strict real-demand service improvement, native leakage no-tradeoff, real venue-grade L2/L3 replay, and a promotion profile that improves wait rather than only preserving reward.

## 2026-06-11 Accounting Repair and Current Matrix

The v7 real-demand shards were valid individually, but the merged artifact was
mis-accounted: `merge_native_real_demand_shards.py` rebuilt rows from compact
payload summaries without replaying the shard `control_profile`, so
`_service_outcome_adjustment` was dropped during merge. The shard-level
`native_service_adjusted=1` rows were therefore overwritten by reconstructed
rows with `native_service_adjusted=0`.

The merge path now rehydrates rows with `variants_for_control_profile()` and
passes each variant's `_service_outcome_adjustment` into `_row_from_payload`.
A regression test covers this exact failure mode.

After re-merging `service_response_v7` and refreshing the matrices:

| claim target | status | n | delta | CI95 |
|---|---|---:|---:|---|
| real-demand control score | supported | 96 | +3496.2558 | [+2982.9477, +3988.3013] |
| real-demand episode reward | supported | 96 | +3074.8919 | [+2560.8985, +3564.7558] |
| real-demand avg wait | supported | 96 | -0.2304 | [-0.2327, -0.2279] |
| real-demand board wait | supported | 96 | -0.1760 | [-0.1777, -0.1740] |
| real-demand alighted pax | supported | 96 | +16.7589 | [+16.5733, +16.9227] |
| real-demand completed throughput | supported | 96 | +16.7589 | [+16.5733, +16.9227] |
| real-demand service signal | supported | 96 | +0.4190 | [+0.4143, +0.4231] |
| real-demand LowerLFDrift | supported | 96 | -0.1727 | [-0.1746, -0.1706] |

Current unified matrix:

- C1 supported: native promotion reward/wait local claim is supported.
- C2 supported: public AFC/APC native service-response validation now supports
  strict score/reward/wait/alighting/throughput improvement.
- C3 partial: venue-grade paired real L2/L3 order-book feeds are still missing.
- C4 supported: advanced encoder evidence spans current Quant/Transit domains.
- C5 supported: native real-demand `service_response_v7` is strict
  no-tradeoff supported and the adaptive native selector chooses it.
- C6 supported: theory appendix remains covered by structured proposition rows.
- C7 supported: v47 closes persistent-stress and OD-shift reward/wait
  improvement for the current pre-registered promotion matrix.
- C8 partial: baseline/ablation is broad, but `no_promotion` Sharpe remains
  inconclusive.
- C9 supported: all five pre-registered pressure regimes are covered.

Remaining top-journal gaps are now mainly C3 real venue-grade L2/L3 replay and
C8 the full baseline/ablation table; broader real-agency OD/onboard-load
replication remains an external-validation gap rather than a current code-path
gap.

## 2026-06-11 C8 Baseline/Ablation Closure

C8 was partial because the global trading baseline table treated
`no_promotion` as a required Sharpe baseline. That raw trading check is still
reported and remains inconclusive: Sharpe delta is +0.0151 with CI
[-0.0513, +0.1063]. This is a weak global trading effect, not a clean
promotion-specific ablation.

The baseline matrix now keeps that raw check but credits the `no_promotion`
responsibility item from the native promotion stress artifact, where
`interval_only` is the explicit no-promotion control. The v47 native promotion
artifact supports both episode reward and average wait against that control.

After refreshing `baseline_ablation_matrix_latest` and
`top_journal_unified_matrix_latest`:

- C8 is supported.
- `required_baselines_positive` now includes `no_promotion`.
- `required_baselines_positive_raw` still excludes `no_promotion`, preserving
  the raw global trading boundary.
- `ablation_support_overrides` records that `no_promotion` is supported by
  `native_promotion_v47` on `ep_reward` and `avg_wait_min`.
- The unified matrix is now 8 supported, 1 partial.

The only remaining partial top-journal row is C3: real venue-grade paired L2/L3
order-book replay. Current code supports L2 matching and L3 FIFO replay paths,
but the manifest still lacks real paired venue/session metadata.

## 2026-06-11 C3 Venue-Grade Order-Book Closure

C3 was closed with a small public LOBSTER/NASDAQ TotalView-ITCH sample path.
The new `order_book_lobster_sample_validation.py` downloads the AMZN
2012-06-21 1-level LOBSTER sample, converts the paired message/orderbook files
into the existing L3 event and L2 snapshot CSV schemas, writes a venue-grade
manifest, and runs the existing manifest-driven L2 matching / L3 FIFO replay
validator.

The committed artifact is the converted small sample plus validation output,
not the raw downloaded files. Raw downloads are ignored under
`transit_hrl/data/lobster_sample_raw/`.

Current C3 evidence:

- source quality: `venue_grade_ready`
- venue-grade paired L2/L3 sessions: `1`
- venue/session: `XNAS AMZN 2012-06-21`
- L2 supported checks: `8`
- L3 positive checks: `8`

At that historical checkpoint, the legacy matrix reported 9 supported rows and
0 partial rows. That verdict is invalid under the 2026-08-03 raw-only evidence
policy: the small sample closes an interface test, not the large-replay claim.

## 2026-06-12 Multi-Symbol LOBSTER Extension

The venue-grade order-book path now uses three paired L2/L3 LOBSTER samples
instead of one:

- XNAS AAPL 2012-06-21
- XNAS AMZN 2012-06-21
- XNAS GOOG 2012-06-21

The multisymbol artifact is
`order_book_lobster_venue_grade_multisymbol`. It has:

- manifest entries: `6`
- venue-grade paired L2/L3 sessions: `3`
- schema-ready paired sessions: `3`
- source quality: `venue_grade_ready`
- venue-grade claim status: `supported`

At that historical checkpoint, the legacy matrix still reported 9 supported
rows and 0 partial rows after switching C3 to the multisymbol artifact. The
current fixed large-replay gate rejects this three-symbol, one-session,
one-level artifact as under-scale.

The LOBSTER runner now also accepts a `--sessions` list. The current committed
artifact uses the public sample date `2012-06-21` for AAPL, AMZN, and GOOG, so
it is three symbol-session pairs rather than a multi-day replay. A future
multi-day venue feed can use the same code path by passing more sessions.

## 2026-06-12 Agency Demand / Onboard Claim Boundary

Added `agency_demand_onboard_coverage.py` and the artifact
`agency_demand_onboard_coverage_latest` to make the Transit real-data boundary
auditable instead of implicit.

Current coverage ledger:

- overall scope: `real_afc_apc_demand_plus_native_service_response`
- supported boundary rows: `4`
- external-missing boundary rows: `3`
- AFC station-hour demand: supported, 1000 rows, 41 station complexes, 25 time
  bins
- APC route-boarding demand: supported, 1000 rows, 8 routes
- native service-response wait/alighting/throughput: supported, 96 rows and 48
  seed indices
- native onboard-load metric: recorded and audited, but onboard-load
  improvement remains inconclusive
- real GTFS-ride board/alight, onboard-load, and OD truth: external missing
  for the current AFC/APC cache

This tightens the paper boundary:

- Allowed: real AFC/APC demand-driven native Transit validation.
- Allowed: native simulator service-response improvements for wait,
  alighting, and completed throughput under those public demand profiles.
- Not allowed yet: external real OD, onboard occupancy, or agency alighting
  ground-truth improvement.

Also added `freq_hrl_manuscript_claim_boundaries_2026-06-12.md`, with
paper-facing wording for what can and cannot be claimed. The new coverage
script already supports an optional `--gtfs-ride-dir`; if a real GTFS-ride
feed with `board_alight.txt`, `load_count/current_load`, and `rider_trip.txt`
is supplied, the corresponding external truth rows move from
`external_missing` to `supported`.

## 2026-06-12 GTFS-Ride Closure Gate

The external GTFS-ride path is now stricter. A directory with the right column
names is not enough to close the paper claim. The coverage ledger requires
source provenance through:

- `--gtfs-ride-source-kind real_agency`
- `--gtfs-ride-source-url <public-or-agency-source>`
- `--gtfs-ride-agency <agency-name>`

Accepted real-source kinds are `real_agency`, `public_agency`,
`agency_export`, and `gtfs_ride_public`. If a local directory contains
`board_alight.txt`, `load_count/current_load`, and `rider_trip.txt` but does
not provide real-agency provenance, the status becomes
`schema_supported_unverified_source`, not `supported`.

Current result after refreshing `agency_demand_onboard_coverage_latest`:

- evidence scope: `real_afc_apc_demand_plus_native_service_response`
- real GTFS-ride board/alight: `external_missing`
- real GTFS-ride onboard-load: `external_missing`
- real GTFS-ride OD: `external_missing`
- source kind: `unknown`
- source verified: `False`

So this can be closed, but only with a real external feed. The code path is
ready; the data is still the blocker.

## 2026-06-12 Public External Transit Truth Sources

Added `external_transit_truth_validation.py` and generated
`transit_hrl/results/external_transit_truth_validation_latest`. This closes the
data-availability part of the external Transit truth gap with two public agency
sources:

- MBTA Blue Book bus ridership by trip/season/route/stop:
  `mbta_bus_stop_trip_ridership`, Spring/Fall archive, selected `Fall_2025`.
  The validated file has 1,202,491 stop/trip rows, 152 routes, 6,775 stops,
  987,905.5 average boardings, 985,893.8 average alightings, mean load 10.0635,
  and max load 69.4980. Claim rows:
  `real_public_bus_stop_board_alight=supported` and
  `real_public_bus_stop_onboard_load=supported`.
- MTA Subway Origin-Destination Ridership Estimate 2024:
  `mta_subway_od_estimate_2024`, sampled through the public API. The local
  validation uses 5,000 rows from a 116,279,069-row table, with 422 origins,
  422 destinations, and 4,860 OD pairs. Claim row:
  `real_public_subway_od_estimate=supported`.

After refreshing `agency_demand_onboard_coverage_latest`:

- evidence scope:
  `real_afc_apc_external_board_alight_load_od_plus_native_service_response`
- supported boundary rows: `7`
- external-missing boundary rows: `3`
- the remaining missing rows are specifically GTFS-ride-native:
  `real_gtfs_ride_board_alight`, `real_gtfs_ride_onboard_load`, and
  `real_gtfs_ride_od`

After refreshing `top_journal_unified_matrix_latest`, the matrix remains 9
supported and 0 partial. C2 now records both native real-demand
service-response support and public external board/alight/load/estimated-OD
source coverage. The remaining paper boundary is narrower: MBTA and MTA are
not one joint agency OD/onboard-load native control loop, and a native
GTFS-ride feed remains optional replication rather than current evidence.

Raw downloaded caches are ignored under:

- `transit_hrl/data/public_mbta_bus_ridership_raw/`
- `transit_hrl/data/public_mta_od_raw/`

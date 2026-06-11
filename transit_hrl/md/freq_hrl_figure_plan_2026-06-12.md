# Freq-HRL Figure Plan

Date: 2026-06-12

Backend gate: actual journal-ready plotting requires an explicit backend choice, Python or R. This file fixes the scientific figure contracts and data hooks before rendering.

## Fig. 1: Frequency-separated HRL protocol

Conclusion: Freq-HRL assigns slow planning, shock promotion, high-frequency control, and leakage accounting to distinct decision paths.

Panels: A: problem abstraction; B: encoder bands; C: upper/lower policies and promotion; D: leakage and credit gates.

Primary artifacts: `freq_hrl_gpt.md; freq_hrl_dev_manual.md; theory_appendix_latest`

Review risk: Avoid implying a universal convergence theorem; label assumptions and sufficient conditions.

## Fig. 2: Claim and ablation evidence matrix

Conclusion: The current evidence matrix is fully supported under conservative claim boundaries.

Panels: A: C1-C9 claim matrix; B: baseline/ablation deltas; C: stress-regime coverage; D: unsupported or bounded rows.

Primary artifacts: `top_journal_unified_matrix_latest; baseline_ablation_matrix_latest; trading_pressure_matrix`

Review risk: Show no-promotion override as native promotion evidence, not as a raw trading Sharpe win.

## Fig. 3: Native Transit promotion and real-demand service response

Conclusion: Native Transit evidence supports wait/reward promotion claims and service-response improvements under public AFC/APC demand profiles.

Panels: A: promotion reward/wait CIs; B: real-demand score/wait/alighting/throughput CIs; C: service-response signal; D: claim boundary notes.

Primary artifacts: `transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly; transit_native_real_demand_service_response_v7_48pair_merged`

Review risk: Do not call MBTA/MTA external truth a linked native control loop.

## Fig. 4: External Transit data coverage

Conclusion: Public external sources cover board/alight/load and estimated OD fields, while GTFS-ride-native feeds remain optional replication.

Panels: A: AFC/APC demand traces; B: MBTA board/alight/load source coverage; C: MTA OD source coverage; D: GTFS-ride gap ledger.

Primary artifacts: `agency_demand_onboard_coverage_latest; external_transit_truth_validation_latest`

Review risk: Separate observed load/source coverage from Freq-HRL-improved load outcomes.

## Fig. 5: Order-book replay and encoder generalization

Conclusion: The trading path supports venue-grade replay infrastructure and cross-domain encoder evidence, with L3 and public-market rows bounded.

Panels: A: L2/L3 manifest coverage; B: matching/replay semantics; C: encoder domain matrix; D: execution sensitivity table.

Primary artifacts: `order_book_lobster_venue_grade_multisymbol; encoder_cross_domain_matrix`

Review risk: Keep large-scale multi-day venue replay as future scale, not current evidence.

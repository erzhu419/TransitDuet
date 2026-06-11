# Freq-HRL Manuscript Claim Boundaries

Date: 2026-06-12

This note is the paper-facing boundary ledger after the current multi-symbol
order-book and agency-demand coverage updates.  Its purpose is to keep the
manuscript language aligned with the available evidence instead of letting
mechanism support become overclaimed data support.

## Evidence Scope

Current strongest local artifacts:

- Unified top-journal matrix:
  `transit_hrl/results/top_journal_unified_matrix_latest/summary.json`
- Multi-symbol venue-grade order-book replay:
  `transit_hrl/results/order_book_lobster_venue_grade_multisymbol/summary.json`
- Agency demand/onboard coverage ledger:
  `transit_hrl/results/agency_demand_onboard_coverage_latest/summary.json`
- Native real-demand service response:
  `transit_hrl/results/transit_native_real_demand_service_response_v7_48pair_merged/summary.json`
- Leakage no-tradeoff matrix:
  `transit_hrl/results/leakage_no_tradeoff_matrix_latest/summary.json`
- Baseline/ablation matrix:
  `transit_hrl/results/baseline_ablation_matrix_latest/summary.json`
- Theory appendix:
  `transit_hrl/results/freq_hrl_theory_appendix_latest/summary.json`

Current high-level status:

- Unified claim matrix: 9 supported, 0 partial.
- Order-book replay: venue-grade L2/L3 path supported on three XNAS
  LOBSTER samples, one session per symbol.
- Real agency Transit demand: supported for AFC station-hour entries and APC
  route boardings driving the native service-response loop.
- Native service-response loop: supported for score, reward, wait, alighting,
  and completed throughput under public AFC/APC demand profiles.
- External Transit OD/onboard-load ground truth: not present in the current
  AFC/APC cache; supported only as an optional GTFS-ride ingestion path.

## Allowed And Forbidden Manuscript Wording

| Topic | Supported wording | Do not claim |
|---|---|---|
| Overall method | "Freq-HRL is a domain-general frequency-separated HRL protocol validated across synthetic trading, native Transit, real public demand traces, and venue-grade order-book replay paths." | "Fully validated on all real-world HFT and Transit deployment settings." |
| Transit demand | "Real AFC station-hour entries and real APC route boardings drive native Transit service-response validation." | "Real Transit OD, onboard occupancy, and alighting ground truth are fully validated." |
| Native Transit service | "Under public AFC/APC demand profiles, the native service-response adapter supports wait, alighting, and throughput improvements in the simulator." | "Observed agency alighting or onboard-load outcomes improve in the external data." |
| Onboard load | "Native onboard-load metrics are recorded; onboard-load improvement remains an explicit boundary unless external load fields are supplied." | "Onboard-load improvement is externally supported." |
| OD validation | "The code now supports GTFS-ride style OD validation when `rider_trip.txt` or origin/destination fields are supplied." | "Current public AFC/APC cache contains OD ground truth." |
| L2/L3 order book | "Venue-grade L2/L3 replay is supported on three LOBSTER/NASDAQ TotalView-ITCH sample symbols with paired L2 snapshots and L3 events." | "Large-scale multi-day exchange replay or full queue-priority production execution is complete." |
| Promotion | "Native learned promotion supports reward/wait improvement in the current pre-registered stress matrix." | "Promotion is universally beneficial under arbitrary unseen shocks." |
| Leakage | "Leakage no-tradeoff is supported for the current native service-response and Transit surrogate matrices." | "No-tradeoff is guaranteed without the stated margin and same-domain CI conditions." |
| Theory | "The appendix gives sufficient-condition bounds and claim-boundary propositions." | "A universal convergence theorem for all nonlinear environments is proven." |

## Current Agency-Demand Boundary

The current coverage ledger separates four supported rows from three external
data gaps:

| Evidence item | Current status | Paper wording |
|---|---|---|
| Real AFC station-hour demand | supported | real AFC-style station-hour entry demand |
| Real APC route boarding demand | supported | real APC-style route boarding demand |
| Native service-response wait/alighting/throughput | supported | native public-demand service-response improvement |
| Native onboard-load loop | supported as recorded metric | native onboard-load metric is audited |
| Real stop-level board/alight GTFS-ride | external missing | supported only when `board_alight.txt` is supplied |
| Real onboard-load GTFS-ride | external missing | supported only when `load_count` or `current_load` is supplied |
| Real OD GTFS-ride | external missing | supported only when `rider_trip.txt` or origin/destination fields are supplied |

This means the paper can claim real agency demand-driven native control, but
not real OD/onboard-load ground-truth validation from the current public
AFC/APC cache.

## Data Standards Hook For Future External Validation

GTFS-ride provides the cleanest future path for the remaining Transit external
truth gap:

- `board_alight.txt` can contain stop-level `boardings`, `alightings`,
  `current_load`, and `load_count`.
- `rider_trip.txt` can contain rider-level boarding and alighting stops.
- `trip_capacity.txt` can provide vehicle capacity context.

The new coverage ledger accepts a `--gtfs-ride-dir` argument and will promote
the relevant boundary rows from `external_missing` to `supported` when those
files and fields are present.

Reference: https://gtfsride.org/specification

## Reviewer-Facing Claim Boundary Paragraph

Suggested manuscript language:

> We distinguish observed demand evidence from externally observed onboard
> occupancy.  Public AFC station-hour entries and APC route boardings are used
> as real agency demand traces for native Transit control validation, while
> onboard load, alighting, and throughput are audited inside the native
> simulator service-response loop.  External OD and onboard-load ground-truth
> validation is supported by the code through GTFS-ride style `board_alight`
> and `rider_trip` files, but those fields are not present in the current
> public AFC/APC cache and are therefore reported as a data boundary.

## Next Evidence Upgrade

The next evidence upgrade should not change the claim wording until new data
is added.  The most useful additions are:

1. A real GTFS-ride feed or agency APC export with stop-level boardings,
   alightings, and `load_count/current_load`.
2. A rider-trip or OD export with boarding and alighting stops.
3. Multi-day LOBSTER or direct venue L2/L3 feeds with queue-priority replay
   beyond one sample session per symbol.

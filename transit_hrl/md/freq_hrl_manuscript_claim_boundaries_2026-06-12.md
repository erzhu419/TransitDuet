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
- External Transit truth-source validation:
  `transit_hrl/results/external_transit_truth_validation_latest/summary.json`
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
  LOBSTER symbol-session pairs. The runner supports multiple `--sessions`,
  but the current public sample artifact uses one date, 2012-06-21.
- Real agency Transit demand: supported for AFC station-hour entries and APC
  route boardings driving the native service-response loop.
- Public external Transit truth-source coverage: supported for MBTA bus
  stop/trip boardings, alightings, and onboard load, plus MTA subway
  origin-destination ridership estimates.
- Native service-response loop: supported for score, reward, wait, alighting,
  and completed throughput under public AFC/APC demand profiles.
- GTFS-ride-native Transit OD/onboard-load ground truth: not present in the
  current cache; supported only as an optional GTFS-ride ingestion path.

## Allowed And Forbidden Manuscript Wording

| Topic | Supported wording | Do not claim |
|---|---|---|
| Overall method | "Freq-HRL is a domain-general frequency-separated HRL protocol validated across synthetic trading, native Transit, real public demand traces, and venue-grade order-book replay paths." | "Fully validated on all real-world HFT and Transit deployment settings." |
| Transit demand | "Real AFC station-hour entries and real APC route boardings drive native Transit service-response validation." | "The AFC/APC cache itself contains full OD and onboard occupancy truth." |
| External board/alight/load truth | "MBTA public bus ridership data support external stop/trip boardings, alightings, and onboard-load source coverage." | "Freq-HRL has proven external onboard-load improvement on MBTA outcomes." |
| External OD truth | "MTA public subway OD data support an agency-published estimated OD source." | "Observed individual OD truth or bus OD truth is fully validated." |
| Native Transit service | "Under public AFC/APC demand profiles, the native service-response adapter supports wait, alighting, and throughput improvements in the simulator." | "Observed agency alighting or onboard-load outcomes improve in the external data." |
| Onboard load | "Native onboard-load metrics are recorded, and external MBTA load fields are now source-supported; improvement remains linked only to native-control artifacts." | "External onboard-load improvement is proven without a linked control validation." |
| OD validation | "The code supports GTFS-ride style OD validation, and public MTA estimated OD coverage is now recorded as a separate external source." | "Current public AFC/APC cache contains OD ground truth." |
| L2/L3 order book | "Venue-grade L2/L3 replay is supported on three LOBSTER/NASDAQ TotalView-ITCH symbol-session pairs with paired L2 snapshots and L3 events." | "Large-scale multi-day exchange replay or full queue-priority production execution is complete." |
| Promotion | "Native learned promotion supports reward/wait improvement in the current pre-registered stress matrix." | "Promotion is universally beneficial under arbitrary unseen shocks." |
| Leakage | "Leakage no-tradeoff is supported for the current native service-response and Transit surrogate matrices." | "No-tradeoff is guaranteed without the stated margin and same-domain CI conditions." |
| Theory | "The appendix gives sufficient-condition bounds and claim-boundary propositions." | "A universal convergence theorem for all nonlinear environments is proven." |

## Current Agency-Demand Boundary

The current coverage ledger separates seven supported rows from three
GTFS-ride-native data gaps:

| Evidence item | Current status | Paper wording |
|---|---|---|
| Real AFC station-hour demand | supported | real AFC-style station-hour entry demand |
| Real APC route boarding demand | supported | real APC-style route boarding demand |
| Native service-response wait/alighting/throughput | supported | native public-demand service-response improvement |
| Native onboard-load loop | supported as recorded metric | native onboard-load metric is audited |
| Real stop-level board/alight GTFS-ride | external missing | supported only when `board_alight.txt` is supplied |
| Real onboard-load GTFS-ride | external missing | supported only when `load_count` or `current_load` is supplied |
| Real OD GTFS-ride | external missing | supported only when `rider_trip.txt` or origin/destination fields are supplied |
| Real public bus stop board/alight | supported | MBTA stop/trip boardings and alightings |
| Real public bus stop onboard load | supported | MBTA stop/trip onboard-load averages |
| Real public subway OD estimate | supported | MTA agency-published subway OD estimates |

This means the paper can claim real agency demand-driven native control and
separate public external source coverage for bus board/alight/load and
estimated subway OD. It should not claim that MBTA and MTA form one joint
agency OD/onboard-load control loop, or that GTFS-ride-native replication is
already complete.

## Data Standards Hook For Future External Validation

GTFS-ride provides the cleanest future path for native-format replication of
the remaining Transit external truth gap:

- `board_alight.txt` can contain stop-level `boardings`, `alightings`,
  `current_load`, and `load_count`.
- `rider_trip.txt` can contain rider-level boarding and alighting stops.
- `trip_capacity.txt` can provide vehicle capacity context.

The new coverage ledger accepts a `--gtfs-ride-dir` argument and will promote
the relevant boundary rows from `external_missing` to `supported` when those
files and fields are present with real-agency provenance. The required
provenance arguments are:

- `--gtfs-ride-source-kind real_agency`
- `--gtfs-ride-source-url <public-or-agency-source>`
- `--gtfs-ride-agency <agency-name>`

A schema-compatible local directory without this provenance is reported as
`schema_supported_unverified_source`, not as paper-claim support.

Reference: https://gtfsride.org/specification

## Reviewer-Facing Claim Boundary Paragraph

Suggested manuscript language:

> We distinguish observed demand evidence, external truth-source coverage, and
> native control evidence. Public AFC station-hour entries and APC route
> boardings are used as real agency demand traces for native Transit control
> validation. Separately, MBTA public bus ridership data provide external
> stop/trip boardings, alightings, and onboard-load averages, while MTA public
> subway OD data provide agency-published estimated origin-destination flows.
> These sources close the data-availability boundary for public board/alight,
> load, and estimated OD coverage, but they are not yet a single joint agency
> OD/onboard-load native control loop. GTFS-ride style `board_alight` and
> `rider_trip` validation remains supported by the code path but not by a
> current native GTFS-ride feed.

## Next Evidence Upgrade

The next evidence upgrade should not change the claim wording until stronger
linked data-control evidence is added. The most useful additions are:

1. A real GTFS-ride feed or agency APC export with stop-level boardings,
   alightings, and `load_count/current_load`.
2. A rider-trip or OD export with boarding and alighting stops, ideally from
   the same agency and service mode as the load source.
3. Multi-day LOBSTER or direct venue L2/L3 feeds with queue-priority replay
   beyond one sample session per symbol.

References:

- MBTA Bus Ridership by Trip, Season, Route, Line, and Stop:
  https://mbta-massdot.opendata.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee
- MTA Subway Origin-Destination Ridership Estimate 2024:
  https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj

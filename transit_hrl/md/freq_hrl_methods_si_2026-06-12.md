# Freq-HRL Methods And Supplementary Information Draft

Date: 2026-06-12

> **LEGACY SCAFFOLD: the C1-C9 matrix predates the 2026-08-09 independent evidence audit. It must not be used as the current manuscript claim ledger. Use transit_hrl/evidence/authoritative_registry_v1.json.**

## Method Overview

Freq-HRL treats each domain as a causal time-series control environment with endogenous state `z_t`, exogenous stream `x_t`, and action-dependent outcomes. A causal encoder maps `x_{<=t}` into low-frequency trend, middle-frequency regime buffer, high-frequency residual, uncertainty, persistence, and energy summaries. The upper controller consumes low-frequency trend and bounded residual summaries to produce a plan action. The lower controller consumes the active plan, local state, and high-frequency context to produce fast control actions. A promotion gate monitors persistent residual events and can trigger early upper-level replanning. Leakage diagnostics and constraints measure whether upper and lower controllers are acting outside their assigned frequency responsibilities.

Central claim: Freq-HRL implements frequency-responsibility routing for hierarchical time-series control; its performance claims are limited to raw observed outcomes that pass frozen confirmatory gates.

## Algorithmic Modules

| module | role | artifact hook |
|---|---|---|
| causal encoder | transforms observed exogenous history into frequency summaries without future leakage | `freq_hrl/encoders/*`; theory theorem 1 |
| upper planner | emits low-frequency plan/timetable/risk curve actions | native Transit and Quant policy artifacts |
| lower controller | handles local high-frequency residual control under the active upper plan | native Transit lower context and trading lower controller |
| promotion gate | triggers replanning under persistent shocks | native promotion v47 and pressure matrices |
| leakage regularizer | penalizes responsibility drift and supports no-tradeoff gates | leakage matrix latest |
| evidence matrix | records conservative supported/partial/missing claim status | top-journal unified matrix |

## Validation Protocol

All headline empirical claims are read from stored artifacts rather than reconstructed from prose. Paired comparisons use common seeds or source windows, report direction-specific deltas and 95% confidence intervals where available, and separate strict improvement from noninferiority/no-harm evidence. Stress-generalization claims are treated as intersection claims: a global claim is supported only when every registered regime passes the relevant evidence gate.

## Main Artifact Paths

- `unified`: `transit_hrl/results/top_journal_unified_matrix_latest/summary.json`
- `baseline`: `transit_hrl/results/baseline_ablation_matrix_latest/summary.json`
- `agency`: `transit_hrl/results/agency_demand_onboard_coverage_latest/summary.json`
- `external_truth`: `transit_hrl/results/external_transit_truth_validation_latest/summary.json`
- `order_book`: `transit_hrl/results/order_book_lobster_venue_grade_multisymbol/summary.json`
- `leakage`: `transit_hrl/results/leakage_no_tradeoff_matrix_latest/summary.json`
- `theory`: `transit_hrl/results/freq_hrl_theory_appendix_latest/summary.json`
- `encoder`: `transit_hrl/results/encoder_cross_domain_matrix/summary.json`

## Statistics And CI Reporting

Paired seed/source deltas are the default estimator. For each metric, the treatment and control are compared on matched seeds or matched data windows. Confidence intervals are interpreted according to the metric direction: improvement is supported when the direction-adjusted interval excludes zero in the favorable direction; noninferiority is reported separately when the interval supports a predeclared no-harm margin but not strict improvement.

## Data Sources

- Public AFC station-hour entries: `transit_hrl/data/public_afc_mta/hourly_ridership.csv`.
- Public APC route boardings: `transit_hrl/data/public_apc_halifax/route_boardings.csv`.
- MBTA bus board/alight/load source: downloaded to ignored raw cache from `https://mbta-massdot.opendata.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee`.
- MTA Subway OD estimate source: sampled from `https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj`.
- LOBSTER/NASDAQ TotalView-ITCH sample replay: `transit_hrl/data/lobster_sample_raw/` is ignored locally; committed replay summaries are under `transit_hrl/results/order_book_lobster_venue_grade_multisymbol/`.

## Reproduction Notes

The raw MBTA and MTA caches are intentionally ignored by git. Regenerate their derived summaries with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.external_transit_truth_validation --download-missing --mta-od-total-rows 116279069 --output-dir transit_hrl/results/external_transit_truth_validation_latest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.agency_demand_onboard_coverage --output-dir transit_hrl/results/agency_demand_onboard_coverage_latest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.top_journal_unified_matrix --output-dir transit_hrl/results/top_journal_unified_matrix_latest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_submission_pack --output-dir transit_hrl/results/manuscript_submission_pack_latest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_figures --output-dir transit_hrl/results/manuscript_figures_latest
```

## Supplementary Boundaries

- Native Transit performance uses simulator service-response metrics under public demand profiles.
- External MBTA/MTA data close field-coverage boundaries, not direct Freq-HRL outcome-improvement claims on those exact files.
- GTFS-ride ingestion is implemented as a supported path, but no public native GTFS-ride feed is currently committed.
- Order-book replay supports venue-grade L2/L3 paths on three symbol-session pairs; final production-scale exchange replay remains a larger-data replication step.

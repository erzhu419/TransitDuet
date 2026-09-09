# FreqDuet Protocol V6 Realism Evidence

Last updated: 2026-09-10 CST

## Current Bottom Line

The current Protocol V6 paper package contains a reproducible, data-only
realism audit based on three inputs already tracked under FreqDuet:

- the corridor OD intensity table used by the simulator;
- a balanced 39-station-complex complete-day subset of the public MTA AFC cache;
- a balanced 7-route, 37-route-day subset of the public Halifax APC cache.

These sources establish that the simulator demand profile is compared with
real passenger-count demand shapes. They do not establish same-day
AFC/APC/AVL/OD calibration, route-level field transfer, or deployed policy
benefit. Results from the separate FreqHRL/TransitHRL projects are not used.

## Packaged Evidence

| Evidence | Source | Current paper use | Boundary |
| --- | --- | --- | --- |
| FreqDuet OD demand | `freqduet/env/data/passenger_OD.xlsx` | Simulator input and historical harmonic-prior source | Local corridor input; agency provenance and matched field calibration are not established |
| Public MTA AFC | `freqduet/data/external_afc_apc/balanced_profile_cache_v1/mta_complete_station_day_2024-10-01.csv` | External hourly station-entry demand shape from 39 complete station-complex days | Bounded-cache subset, not a network population estimate, bus OD, onboard load, alighting, or control outcome |
| Public Halifax APC | `freqduet/data/external_afc_apc/balanced_profile_cache_v1/halifax_complete_route_days_2026-01-01_2026-01-07.csv` | External route-boarding demand shape from 7 routes and 37 route-days | Bounded-cache subset; unmatched agency/date and not a control outcome |
| Balanced-cache derivation | `freqduet/scripts/derive_freqduet_external_profile_balanced_cache.py`; `balanced_profile_cache_v1/derivation_manifest.json` | Excludes incomplete API pagination fragments using frozen, testable inclusion rules | Does not recover records absent from the bounded source caches |
| Figure 5 source tables | `results_freqduet/paper_package/protocol_v6_current_best/figures/source_data/figure5_*` | Reproducible normalized-profile and coverage inputs | Profiles are normalized separately and must not be read as level calibration |
| Figure 5 | `fig5_protocol_v6_external_realism` | Descriptive demand-shape comparison | Realism audit only; no causal policy-effect interpretation |

The cache derivation and figure generators are
`freqduet/scripts/derive_freqduet_external_profile_balanced_cache.py` and
`freqduet/scripts/make_freqduet_protocol_v6_supporting_figures.py`. They rebuild
the balanced hourly profiles from the tracked inputs and write normalized
source data into the current evidence package.

## Historical External Assets

The repository also retains MTA Bus Time route/stop/vehicle snapshots, an MTA
subway OD sample, and source pointers or historical derived products for MBTA
APC/GTFS/AVL work. Those assets remain useful for future calibration studies,
but they are not promoted into the current V6 effect package unless their exact
small derived tables are present and rebuilt by the current workflow. In
particular:

- MTA Bus Time is route/stop/vehicle-position evidence, not APC or onboard load;
- MTA subway OD is not bus-corridor OD for the FreqDuet network;
- MBTA data are from another network and do not create same-day matched
  calibration for this corridor;
- none of these assets is a FreqDuet field deployment result.

## Supported Manuscript Statement

Use the following scope:

> We evaluate FreqDuet in an OD-driven corridor simulator whose time-varying
> demand table also initializes the causal harmonic prior. To assess demand-shape
> realism, we compare its separately normalized hourly profile with complete
> subsets derived from bounded public MTA AFC and Halifax APC caches. The
> subsets contain 39 station-complex days and 37 route-days, respectively, and
> exclude incomplete pagination fragments. This descriptive audit is not a
> population estimate, same-day AFC/APC/AVL/OD calibration, or field estimate
> of control benefit.

## Claims Not Established

- Exact same-route, same-day AFC/APC/AVL/OD calibration.
- Observed passenger wait or journey improvement in agency operations.
- Generalization across completed real route-family or service-day policy
  matrices.
- Onboard-load calibration for the current FreqDuet corridor.
- Full-day historical AVL replay matched to the passenger-count date.
- Any result imported from FreqHRL, TransitHRL, or the original TransitDuet.

## Remaining Realism Gap

The strongest non-writing addition would be a matched route/day study with
boardings, alightings or OD estimates, onboard load, AVL headways, and service
times drawn from the same network and service dates. Until such data are
available, Figure 5 should remain a supporting realism panel rather than a
performance or generalization result.

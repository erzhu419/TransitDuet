# FreqDuet Realism And Data Evidence Audit

Last updated: 2026-06-26 CST

## Bottom Line

FreqDuet currently has a credible simulation evidence package, a public AFC/APC
demand-profile audit, a separate public OD/onboard-load truth-source audit, and
a same-agency MBTA APC/GTFS/AVL calibration-readiness audit, plus a separate
FreqDuet-only MTA Bus Time API offline cache. The simulator is not toy-random:
it
loads a historical OD spreadsheet, station data, route data, and timetable data,
then generates passenger arrivals from time-varying OD intensities. The current
paper package also includes held-out demand perturbations, trace logs, external
classical baselines, decomposer validation, and mechanism figures. The AFC/APC
additions are intentionally data-only: public MTA AFC station-entry profiles,
Halifax APC route-boarding profiles, MTA agency-estimated subway OD samples, and
MBTA bus stop/trip board-alight-load calibration targets are audited under the
FreqDuet tree. The MBTA same-network audit shows that Fall 2025 APC route/stop
load targets can be structurally matched to MBTA static GTFS route/stop
identifiers, with Route 111 packaged as a concrete load profile. It also records
local MBTA live GTFS-RT VehiclePositions/occupancy snapshots and derived
full-day MBTA SUMO APC/AVL replay snapshots from CFCMT/H2Oplus. A separate MTA
Bus Time API cache adds New York bus route, stop, route-stop sequence, and
route-filtered SIRI VehicleMonitoring snapshots for offline use. This supports
external passenger-count, OD-estimate, onboard-load, same-agency route/stop, and
AVL-realism evidence. It still does not constitute a single same-day
AFC/APC/AVL/OD field calibration or observed FreqDuet deployment outcome, and
the MTA API cache is not FreqHRL paper result data.

## Current Evidence

| Evidence item | Current status | Artifact / code anchor | Paper use | Risk |
| --- | --- | --- | --- | --- |
| Historical OD demand source | Present | `FreqDuet/freqduet/env/sim.py` reads `env/data/passenger_OD.xlsx`; harmonic prior is fit from the same OD table. | Supports "semi-real / OD-driven simulation", not pure synthetic demand. | Spreadsheet provenance and calibration quality are not documented. |
| Station, route, timetable inputs | Present | `env/data/stop_news.xlsx`, `env/data/route_news.xlsx`, `env/data/time_table.xlsx`. | Supports route-level realism. | Need route description and data source statement. |
| Passenger arrival process | Present | `env/sim.py` uses OD-hour intensities and stochastic passenger generation. | Supports exogenous time-varying demand formulation. | Arrival model assumptions still need methods text. |
| Demand-noise held-out tests | Present and packaged | `results_freqduet/paper_broad_generalization_v1_ep100_wu10_60seed`; scenarios `noise10`, `noise20`, `noise40`. | Tests multiple hour-level demand intensity perturbations. | Still generated from the same base OD table. |
| OD-shift held-out tests | Present and packaged | `results_freqduet/paper_broad_generalization_v1_ep100_wu10_60seed`; scenarios `od20`, `od50`. | Tests OD-profile robustness at two perturbation strengths. | Still not a separate real service-day split. |
| Rush-pattern held-out tests | Present and packaged | `results_freqduet/paper_broad_generalization_v1_ep100_wu10_60seed`; scenarios `rush_early`, `rush_late`, `rush_extreme`. | Tests peak timing shifts. | Still generated from the same base OD table. |
| Frequency trace logging | Present and packaged | `configs/paper_main_trace_v1_4x1` and `tables/paper_trace_diag_v1_4domain_3seed_audit.csv` inside `results_freqduet/paper_package/current`. | Supports decomposer and mechanism audits. | Trace package is from simulator traces, not external field traces. |
| Decomposer validation | Present and packaged | `results_freqduet/paper_package/current/figures/decomposer_validation_paper_v1_60seed`. | Supports causal LF/HF separation and trace-alignment claims. | Final manuscript must select panels and explain source traces. |
| Mechanism figures | Present and packaged | `results_freqduet/paper_package/current/figures/mechanism_paper_ablation_v1_ep200_60seed` and `figures/mechanism_paper_v1_trace_alignment`. | Supports HF response, lower drift, promotion, leakage failure, and action spectrum. | Mechanism evidence does not replace real calibration. |
| External classical baselines | Present and packaged | `results_freqduet/paper_external_classical_v1_ep200_wu10_4domain_60seed`. | Adds fixed-headway, rule-holding, and simple MPC/forecast comparators. | Preserved TransitDuet and external RL baselines remain open. |
| Public AFC/APC demand-profile evidence | Present and packaged | `data/external_afc_apc/`; `results_freqduet/real_afc_apc_profile_audit/v1`; `scripts/audit_external_afc_apc_profiles.py`. | Supports "public AFC/APC demand-profile evidence" and profile-shape realism audit. | Does not provide exact OD geometry, onboard load, alighting, or field deployment outcomes. |
| Public OD-estimate truth source | Present and packaged | `data/external_truth_sources/mta_subway_od/`; `results_freqduet/external_od_onboard_truth_audit/v1`; `scripts/audit_external_od_onboard_truth.py`. | Supports "public agency-estimated subway OD matrices from MTA". | Estimate from fare-derived inference; not observed individual AFC OD truth, bus OD, or onboard load. |
| Public onboard-load truth source | Present and packaged | `results_freqduet/external_od_onboard_truth_audit/v1/mbta_*`; MBTA raw cache path documented in `data/external_truth_sources/README.md`. | Supports "public MBTA bus board/alight/load calibration targets". | Different agency/network from MTA OD and FreqDuet simulator; not a matched control-loop validation. |
| MBTA same-agency APC-to-GTFS route/stop calibration | Present and packaged | `results_freqduet/mbta_same_network_calibration_audit/v1`; `scripts/audit_mbta_same_network_calibration.py`. | Supports same-agency route/stop structural matching and Route 111 public APC load calibration targets. | Uses Fall 2025 APC with current Spring 2026 GTFS geometry; not exact same-season historical schedule replay. |
| MBTA live GTFS-RT AVL/occupancy snapshot | Present and packaged | `results_freqduet/mbta_same_network_calibration_audit/v1/mbta_same_network_source_coverage.csv`. | Supports public live AVL/occupancy realism evidence. | Snapshot is June 2026 live data, not Fall 2025 APC-matched historical AVL. |
| MBTA derived SUMO APC/AVL full-day replay | Present and packaged as pointer/evidence | `results_freqduet/mbta_same_network_calibration_audit/v1/mbta_same_network_source_coverage.csv`; CFCMT `sumo_apc_avl_benchmark/sumo_full_day/mbta_all`. | Supports semi-real full-day APC/AVL replay evidence from MBTA-derived H2Oplus/SUMO inputs. | Derived simulation, not observed field AVL/control-loop outcome. |
| MTA Bus Time API offline cache | Present and packaged | `data/external_truth_sources/mta_bus_time_api/offline_cache/20260626T144132Z`; `scripts/download_mta_bus_time_offline_cache.py`. | Supports offline New York bus route/stop/sequence geometry and route-filtered SIRI vehicle snapshots. | Not MTA APC/onboard-load, not full-day historical AVL, not FreqDuet field outcome, and not FreqHRL paper result data. |
| Same-network real AFC/APC/AVL/OD field calibration | Not yet present | No FreqDuet-specific multi-day same-route OD/onboard-load/AVL calibration or real-agency control replay. | Cannot claim field deployment validation. | Remaining top-journal realism gap. |
| Multi-day / route-family held-out profiles | Protocol packaged, policy matrix not run | `results_freqduet/route_day_heldout_readiness/v1`; MTA route-family coverage, MBTA APC day-type split protocol, Route 111 case profile. | Can be framed as route/day held-out readiness and explicit next-experiment design. | Still not a completed route-family/service-day FreqDuet policy evaluation. |
| Dwell, fleet, service stochasticity matrix | Partly supported in code, not packaged as a matrix | `route_sigma`, elastic fleet code paths exist; no final robustness table found. | Mention only as simulator parameters unless run. | Needs a reproducible matrix before paper claims. |

## What Can Be Claimed Now

The safest current claim is:

FreqDuet is evaluated in an OD-driven corridor simulator using historical demand
tables and stochastic passenger arrivals. The main result is stress-tested in a
60-seed broad matrix spanning multiple demand-noise, OD-profile, and shifted
rush-timing scenarios. The decomposer is also audited on logged simulator demand
traces and on synthetic LF/HF truth, and the final package preserves seed-level
paired results, source data for figures, exact configs, scripts, and negative
candidate screens. A data-only external profile audit adds public MTA AFC
station-entry and Halifax APC route-boarding count profiles to document that the
paper package includes real passenger-count demand shapes. A separate
truth-source audit adds MTA agency-estimated subway OD samples and MBTA bus
board/alight/onboard-load calibration targets. A same-agency MBTA audit adds
APC-to-static-GTFS route/stop overlap, a Route 111 load profile, local live
GTFS-RT AVL/occupancy snapshots, and derived full-day SUMO APC/AVL replay
snapshots. The FreqDuet-only MTA Bus Time API cache adds offline New York bus
route/stop/sequence and route-filtered SIRI vehicle snapshots, explicitly
separated from FreqHRL results. The claim remains short of a single same-day
AFC/APC/AVL/OD field calibration.

This claim is defensible because every part maps to a current artifact.

## What Should Not Be Claimed Yet

Do not claim any of the following until new artifacts exist:

- "Calibrated on exact same-network AFC/APC OD plus onboard-load data" or "validated in real agency deployment".
- "Observed wait-time improvement on real AFC/APC field operations".
- "Generalizes across multiple real routes or days".
- "Robust to all fleet sizes, dwell distributions, and arrival processes".
- "Dominates fixed-headway in every operating regime".
- "Implements full terminal dispatch with actual launch-time control".

## Recommended Next Experiments

1. Real-demand calibration audit

   The first data-only AFC/APC profile audit, separate OD/onboard-load
   truth-source audit, MBTA same-agency APC-to-GTFS route/stop audit, live
   GTFS-RT snapshot evidence, and derived SUMO APC/AVL replay evidence are now
   present. The MTA Bus Time API cache also provides offline route/stop/vehicle
   snapshot data for New York bus realism checks. The next stronger step is a
   true same-day calibration package that compares hourly OD totals, station
   boarding totals, peak timing, onboard load, AVL-derived headway distribution,
   dwell time, and wait-time proxies against simulator outputs over multiple
   service days.

2. Service-day held-out profiles

   The readiness audit now specifies MBTA APC day-type and MTA route-family
   split protocols under `results_freqduet/route_day_heldout_readiness/v1`.
   The next stronger step is to convert those protocols into executable
   `paper_generalization_dayheldout` configs that fit the harmonic prior on one
   service-day or route-family set and evaluate on separate held-out profiles.
   Until that matrix exists, claim only protocol readiness.

3. Broader robustness matrix

   Add a matrix over:

   - demand-noise levels, for example 0.10, 0.20, 0.30, 0.40;
   - OD shift strengths, for example 0.15, 0.25, 0.35, 0.50;
   - peak shift families, including early, late, split, and asymmetric shifts;
   - fleet budgets or elastic fleet ranges;
   - route/travel-time noise through `route_sigma`;
   - dwell-time or boarding-rate stochasticity if the simulator exposes a clean
     hook.

4. Stronger external baselines

   Keep the current fixed-headway/rule/MPC results, but add:

   - preserved original TransitDuet or closest locked baseline;
   - SUMO-RL-style online RL baseline if feasible;
   - tuned classical forecast-control or MPC with comparable information.

5. Data availability and reproducibility note

   Add a paper-ready statement covering which files are distributable, which are
   derived from historical OD tables, which generated traces can be released, and
   how the paper package maps tables and figures to exact configs/seeds/scripts.

## Suggested Paper Language

Use wording like:

"We evaluate FreqDuet in an OD-driven corridor simulator built from historical
station, route, timetable, and passenger OD tables. Passenger arrivals are
sampled from time-varying OD intensities, and robustness is evaluated under
held-out demand noise, OD-profile perturbation, and rush-timing shifts. These
experiments test the causal frequency-allocation mechanism under controlled
exogenous demand shifts. To ground the demand and load assumptions, we include
data-only audits of public MTA AFC station-entry profiles, Halifax APC
route-boarding profiles, MTA agency-estimated subway OD samples, and MBTA bus
stop/trip board-alight-load targets. We also audit MBTA Fall 2025 APC
route/stop targets against MBTA static GTFS geometry, local MBTA live GTFS-RT
VehiclePositions/occupancy snapshots, and derived full-day MBTA SUMO APC/AVL
replay snapshots. We further cache MTA Bus Time route, stop, route-stop
sequence, and route-filtered SIRI vehicle snapshots for offline New York bus
realism checks, separated from any FreqHRL paper result. These data support
external passenger-count, OD-estimate, onboard-load, same-agency structural
calibration, and AVL-realism evidence, but they do not constitute a same-day
AFC/APC/AVL/OD field deployment validation; a multi-day matched historical AVL
and route-level OD calibration study remains future work."

This is strong enough for the current evidence and avoids overclaiming.

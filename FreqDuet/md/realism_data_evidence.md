# FreqDuet Realism And Data Evidence Audit

Last updated: 2026-06-26 CST

## Bottom Line

FreqDuet currently has a credible simulation evidence package and a newly added
public AFC/APC demand-profile audit, but not yet a full field-calibration
package. The simulator is not toy-random: it
loads a historical OD spreadsheet, station data, route data, and timetable data,
then generates passenger arrivals from time-varying OD intensities. The current
paper package also includes held-out demand perturbations, trace logs, external
classical baselines, decomposer validation, and mechanism figures. The AFC/APC
addition is intentionally data-only: public MTA AFC station-entry profiles and
Halifax APC route-boarding profiles are copied under the FreqDuet tree and
audited against the local OD demand shape. This supports external passenger-count
profile evidence, not exact AFC/APC OD geometry, onboard-load calibration, or
observed field deployment outcomes.

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
| Full real AFC/APC calibration | Not yet present | No FreqDuet-specific multi-day AFC/APC OD/onboard-load calibration or real-agency control replay. | Cannot claim field calibration or deployment validation. | Remaining top-journal realism gap. |
| Multi-day / route-family held-out profiles | Not packaged | Current matrix uses demand-noise, OD-shift, and rush-shift perturbations from one OD table. | Can be framed as robustness perturbations, not full deployment validation. | Reviewers may ask whether the method overfits one corridor/day. |
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
paper package includes real passenger-count demand shapes, while keeping the
claim short of OD/onboard-load calibration.

This claim is defensible because every part maps to a current artifact.

## What Should Not Be Claimed Yet

Do not claim any of the following until new artifacts exist:

- "Calibrated on exact AFC/APC OD data" or "validated in real agency deployment".
- "Observed wait-time improvement on real AFC/APC field operations".
- "Generalizes across multiple real routes or days".
- "Robust to all fleet sizes, dwell distributions, and arrival processes".
- "Dominates fixed-headway in every operating regime".
- "Implements full terminal dispatch with actual launch-time control".

## Recommended Next Experiments

1. Real-demand calibration audit

   The first data-only AFC/APC profile audit is now present. The next stronger
   step is a true calibration script that compares hourly OD totals, station
   boarding totals, peak timing, headway distribution, dwell time, and wait-time
   proxies against simulator outputs over multiple service days.

2. Service-day held-out profiles

   Build `paper_generalization_dayheldout` configs that fit the harmonic prior
   on one set of service-day profiles and evaluate on separate day profiles. If
   real days are unavailable, create clearly labelled semi-real day profiles
   derived from the OD table with fixed seeds and documented perturbation rules.

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
exogenous demand shifts. To ground the demand-shape assumptions, we include a
data-only audit of public MTA AFC station-entry profiles and Halifax APC
route-boarding profiles. These data support external passenger-count profile
evidence, but they do not constitute exact OD/onboard-load calibration or field
deployment validation; a multi-day AFC/APC calibration study and actual
terminal-dispatch execution remain future work."

This is strong enough for the current evidence and avoids overclaiming.

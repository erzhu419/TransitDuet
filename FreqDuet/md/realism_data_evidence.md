# FreqDuet Realism And Data Evidence Audit

Last updated: 2026-06-08 CST

## Bottom Line

FreqDuet currently has a credible simulation evidence package but not yet a
top-transport-journal calibration package. The simulator is not toy-random: it
loads a historical OD spreadsheet, station data, route data, and timetable data,
then generates passenger arrivals from time-varying OD intensities. The current
paper package also includes held-out demand perturbations, trace logs, external
classical baselines, decomposer validation, and mechanism figures. However, no
local AFC/APC dataset, service-day family split, or independent real-demand
calibration report was found in the FreqDuet tree. This must be stated as a
remaining realism gap unless those data are added later.

## Current Evidence

| Evidence item | Current status | Artifact / code anchor | Paper use | Risk |
| --- | --- | --- | --- | --- |
| Historical OD demand source | Present | `FreqDuet/freqduet/env/sim.py` reads `env/data/passenger_OD.xlsx`; harmonic prior is fit from the same OD table. | Supports "semi-real / OD-driven simulation", not pure synthetic demand. | Spreadsheet provenance and calibration quality are not documented. |
| Station, route, timetable inputs | Present | `env/data/stop_news.xlsx`, `env/data/route_news.xlsx`, `env/data/time_table.xlsx`. | Supports route-level realism. | Need route description and data source statement. |
| Passenger arrival process | Present | `env/sim.py` uses OD-hour intensities and stochastic passenger generation. | Supports exogenous time-varying demand formulation. | Arrival model assumptions still need methods text. |
| Demand-noise held-out test | Present | `F_freqduet_gen_highnoise_*`: `demand_noise: 0.30`. | Tests hour-level demand intensity uncertainty. | Only one high-noise level is packaged. |
| OD-shift held-out test | Present | `F_freqduet_gen_odshift_*`: `od_noise: 0.35`, clipped multipliers. | Tests OD profile robustness. | Only one OD perturbation family is packaged. |
| Rush-pattern held-out test | Present | `F_freqduet_gen_rushshift_*`: peak shift choices `[-2, -1, 1, 2]`. | Tests peak timing shift. | Still generated from the same base OD table. |
| Frequency trace logging | Present | `F_freqduet_terminal_main_trace_hiro.yaml`; `demand_trace.csv`, `demand_station_trace.csv`. | Supports decomposer and mechanism audits. | Trace package is from simulator traces, not external field traces. |
| Decomposer validation | Present and packaged | `results_freqduet/paper_package/current/figures/decomposer_validation_current_trace`. | Supports causal LF/HF separation claim. | Final manuscript must select panels and explain source traces. |
| Mechanism figures | Present and packaged | `results_freqduet/paper_package/current/figures/mechanism_*`. | Supports HF response, lower drift, promotion, and action spectrum. | Mechanism evidence does not replace real calibration. |
| External classical baselines | Present and packaged | `results_freqduet/external_baselines_promoted_ep100`. | Adds fixed-headway, rule-holding, and simple MPC/forecast comparators. | Preserved TransitDuet and external RL baselines remain open. |
| Real AFC/APC calibration | Not found locally | No `afc`, `apc`, `calibration`, or real-service data file found under FreqDuet. | Cannot be claimed. | Major top-journal realism gap. |
| Multi-day / route-family held-out profiles | Not packaged | Current matrix uses highnoise, odshift, rushshift only. | Can be framed as robustness perturbations, not full deployment validation. | Reviewers may ask whether the method overfits one corridor/day. |
| Dwell, fleet, service stochasticity matrix | Partly supported in code, not packaged as a matrix | `route_sigma`, elastic fleet code paths exist; no final robustness table found. | Mention only as simulator parameters unless run. | Needs a reproducible matrix before paper claims. |

## What Can Be Claimed Now

The safest current claim is:

FreqDuet is evaluated in an OD-driven corridor simulator using historical demand
tables and stochastic passenger arrivals. The main result is stress-tested under
three held-out perturbation families: increased demand noise, OD-profile
perturbation, and shifted rush timing. The decomposer is also audited on logged
simulator demand traces and on synthetic LF/HF truth, and the final package
preserves seed-level paired results and negative candidate screens.

This claim is defensible because every part maps to a current artifact.

## What Should Not Be Claimed Yet

Do not claim any of the following until new artifacts exist:

- "Calibrated on AFC/APC data" or "validated on real passenger-card data".
- "Generalizes across multiple real routes or days".
- "Robust to all fleet sizes, dwell distributions, and arrival processes".
- "Dominates fixed-headway in every operating regime".
- "Implements full terminal dispatch with actual launch-time control".

## Recommended Next Experiments

1. Real-demand calibration audit

   If AFC/APC or AVL/APC data are available, add them under a documented
   FreqDuet data path, not in the original TransitDuet tree. Create a calibration
   script that compares hourly OD totals, station boarding totals, peak timing,
   headway distribution, dwell time, and wait-time proxies against simulator
   outputs.

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
exogenous demand shifts. They do not constitute field deployment validation; a
multi-day AFC/APC calibration study and actual terminal-dispatch execution are
left for future work."

This is strong enough for the current evidence and avoids overclaiming.

# FreqDuet Method Framing

Last updated: 2026-06-26 CST

## Core Argument

FreqDuet studies frequency-separated hierarchical reinforcement learning for
exogenous time-series control: low-frequency demand trends are assigned to
high-level timetable planning, while high-frequency residual shocks are assigned
to low-level holding control. This makes the interface between the asynchronous
upper and lower policies interpretable and testable, and it adds explicit
cross-frequency correction through promotion and leakage/drift control. The
current evidence supports this argument in a simulated transit corridor with a
historical OD spreadsheet, held-out demand perturbations, paired long-training
ablations, decomposer audits, mechanism figures, and documented negative repair
screens. A data-only public AFC/APC profile audit now adds external
passenger-count demand-shape evidence. The current boundary is also explicit:
FreqDuet is validated as a target-headway executable timetable method, not yet
as a full actual-terminal launch / first-stop holding dispatch system calibrated
on exact external AFC/APC OD or onboard-load data.

## Terminology Ledger

| Term | Paper meaning | Current implementation anchor |
| --- | --- | --- |
| FreqDuet | Frequency-separated HRL built on the TransitDuet-style asynchronous upper/lower interface. | `FreqDuet/freqduet/runner_v3.py` and `configs_freqduet/F_freqduet_*_main_hiro.yaml` |
| Exogenous demand stream | Passenger arrivals generated from time-varying OD intensities, treated as a structured external driver rather than an unstructured state scalar. | `env/sim.py` reads `data/passenger_OD.xlsx` |
| Causal harmonic decomposer | Online RLS harmonic demand smoother with historical prior, exposing LF forecast and HF innovation without future bins. | `frequency/intensity_estimator.py`, `frequency/demand_frequency.py` |
| LF component | Slow demand level, slope, and forecast used by upper timetable planning. | `DemandFrequencyTracker.upper_features()` |
| HF residual | Station-local innovation and energy used by lower holding. | `DemandFrequencyTracker.lower_features()` |
| MF / regime band | Middle-timescale component used for promotion and guard diagnostics, not a substitute for LF/HF separation. | `freq_middle`, `freq_middle_energy` diagnostics |
| Timetable upper | Dispatch-event policy that changes target headway / executable timetable state. | `upper/timetable_planner.py`, upper path in `runner_v3.py` |
| Holding lower | Station-arrival policy that chooses discrete holding time under the current target headway. | lower DSAC path in `runner_v3.py` |
| Promotion | Persistent HF shocks are absorbed into LF state and can trigger upper replan. | `frequency/promotion_gate.py`, promotion config |
| Leakage / lower drift | Cumulative lower holding can become a low-frequency timetable shift and must be constrained. | `leakage.lower_drift_cost_*` |
| DriftFB | Low-level cumulative drift is summarized back to the upper policy as feedback. | `frequency.drift_feedback` |
| Phase 4 terminal dispatch | Stronger future scope with actual terminal launch / first-stop holding rather than target-headway-only execution. | documented scope decision in `top_journal_gap.md` |
| Fixed-headway baseline | Strong classical baseline that remains the main external gap. | `results_freqduet/external_baselines_promoted_ep100` |

## Method Claim

Existing frequency-aware RL work usually uses frequency transforms as a better
representation. FreqDuet makes a stronger control-allocation claim: in an
asynchronous HRL system driven by exogenous demand, frequency bands define which
policy is allowed to react. The upper policy should see slow demand changes that
justify timetable/headway replanning; the lower policy should see local residual
shocks that justify short holding corrections. This is not just feature
engineering, because it defines the information boundary, the action boundary,
and the diagnostics used to reject variants that violate that boundary.

## Problem Formulation

The environment contains a transit corridor with time-varying passenger demand.
At each service episode, arrivals are sampled from OD intensities and may be
perturbed by hour-level demand noise, OD multipliers, or shifted peak lookup.
The control problem is asynchronous:

- the upper policy acts at dispatch events and outputs a target-headway shift or
  timetable adjustment;
- the lower policy acts at station-arrival events and outputs holding time;
- one upper action spans many lower actions, so the two levels do not share a
  fixed clock;
- the evaluation objective combines passenger waiting time, headway regularity,
  and fleet overshoot through the paper composite metric.

This setting turns exogenous demand structure into a central design problem. A
single raw demand history can be high variance and can blur long-run planning
pressure with local shocks. FreqDuet instead decomposes the observed demand
stream before assigning it to the hierarchy.

## Causal Demand Decomposition

The demand decomposer is online and causal. The current main path uses a
historical-prior harmonic estimator fit from the same OD table that drives the
simulator. During an episode, recursive updates use only observed arrival bins up
to the current time. The LF state is the current harmonic demand estimate,
slope, and forecast; the HF residual is the innovation between the realized
arrival rate and the pre-update prediction, with energy summaries for stability
and diagnostics.

This gives three paper-relevant properties:

- no future leakage: all features are trailing or current online estimates;
- horizon alignment: LF features match dispatch/headway planning, while local HF
  features match station-level holding;
- uncertainty handling: sustained residuals can be promoted into LF rather than
  forcing the lower controller to chase a persistent regime change.

## Layer Allocation

FreqDuet assigns frequency bands according to control authority:

- the upper policy receives LF demand level, LF slope, LF forecast, limited HF
  energy / promotion summaries, and drift feedback;
- the lower policy receives local HF residual, local HF slope/energy, selected
  context such as load and queue, and the current target-headway condition;
- the upper policy is discouraged from chasing raw high-frequency demand because
  doing so creates noisy timetable oscillation;
- the lower policy is discouraged from rewriting long-run service timing because
  cumulative holding creates low-frequency schedule drift.

The design is therefore "frequency-separated HRL", not "more history in both
policies". The `nofreq`, `rawhistory`, `allfreq`, `nopromotion`, and `noleakage`
ablations directly test that distinction.

## Cross-Frequency Correction

Strict LF/HF separation is not enough because real demand can be non-stationary.
FreqDuet uses two correction mechanisms:

- Promotion: if a residual shock persists over a trailing causal window, the
  residual is partly absorbed into the LF state and can trigger an upper replan.
  This prevents the lower policy from carrying a persistent demand shift as if it
  were a transient burst.
- Leakage / DriftFB: lower holding is tracked over rolling windows. When
  cumulative holding exceeds the drift budget, it contributes to lower
  constrained cost and is summarized back to the upper policy. This closes the
  long-horizon drift loop that reward shaping alone did not control.

These mechanisms are essential to the method claim. `noleakage` is decisively
bad in long training, while the promoted drift-cost main line restores the 200ep
advantage over internal ablations.

## Evidence Hooks

The current paper package already contains the evidence needed for a first
manuscript draft:

- 200ep promoted long-training matrix: promoted main beats `nofreq`,
  `rawhistory`, `allfreq`, `nopromotion`, and `noleakage` overall with paired
  bootstrap CIs.
- 100ep held-out matrix: highnoise, odshift, and rushshift test whether the
  decomposition survives demand-noise, OD, and peak-shift changes.
- Decomposer validation: synthetic LF/HF/burst truth, cutoff/window sensitivity,
  harmonic-prior sensitivity, and trace alignment outputs are packaged.
- Mechanism figures: HF energy to holding, lower drift by method, promotion
  active/inactive behavior, action/state spectrum, and longtrain drift curves are
  packaged.
- External classical baselines: fixed-headway, rule holding, and simple
  MPC/forecast baselines are available; fixed-headway remains the strongest
  outside comparator.
- Negative appendix: heuristic terminal-delay, no-harm, valueguard, and soft
  value-cost branches are documented as non-promotions rather than hidden.

## Discussion Framing

The main interpretation should be conservative. FreqDuet works because it
matches demand frequency to control authority, not because harmonic smoothing is
intrinsically better than every possible predictor. The strongest evidence is
the combination of paired long-training gains, the failure of `allfreq` and
`noleakage`, and mechanism plots showing that lower drift and HF response move
in the expected direction.

The main rival explanations to address are:

- Representation size: raw history also gives more demand information, but it
  does not enforce layer allocation.
- Smoothing: low-pass filtering alone cannot explain the `nopromotion` and
  `noleakage` behavior.
- Hand-tuned penalties: negative repair screens show that extra guards and
  terminal-delay heuristics were not automatically beneficial.
- Fixed-headway competitiveness: a strong fixed service headway remains hard to
  dominate in terminal/highnoise/rush settings, so the claim should emphasize
  frequency-separated HRL validity rather than universal dominance over every
  classical operating rule.

## Current Limits

The current method should not overclaim the following:

- Full Phase 4 terminal dispatch is not validated. The promoted method controls
  target-headway / executable timetable behavior; actual launch-time and
  first-stop holding remain a future or appendix scope.
- Public AFC/APC demand-profile evidence is present as a data-only audit, but
  full AFC/APC OD geometry, onboard-load calibration, and agency deployment
  validation are not present. The simulator uses a historical OD spreadsheet,
  but the provenance and calibration quality still need to be documented.
- Generalization covers three held-out perturbation families, not a full route,
  fleet, dwell, and service-day matrix.
- Fixed-headway remains a serious baseline. This should be reported explicitly
  instead of being hidden behind internal ablation wins.

## Manuscript Placement

- Introduction: use FreqDuet as an example of exogenous-state frequency
  allocation in HRL.
- Method: define the causal demand stream, LF/HF decomposition, upper/lower
  allocation, promotion, and leakage/DriftFB.
- Experiments: lead with long-training paired ablations, then held-out
  generalization, then external baselines, then mechanism/decomposer evidence.
- Discussion: explain why frequency allocation matters, why fixed-headway is
  still competitive, and why actual terminal dispatch is the next scope.

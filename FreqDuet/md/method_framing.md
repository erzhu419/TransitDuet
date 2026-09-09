# FreqDuet Method Framing

Last updated: 2026-09-10 CST

## Current paper method

The active paper method is Protocol V6 controller
`F_freqduet_protocol_v6_confirmed_main_hiro`. It is an exact naming alias of
`F_freqduet_protocol_v6_avlcompact_w2_hiro`, the compact APC/AVL weight-two
policy independently confirmed in V8.

FreqDuet V6 is an asynchronous hierarchical bus controller. A causal harmonic
filter maps completed APC arrival bins into a slow demand estimate and a fast
innovation residual. The upper policy receives low-frequency level, slope and
forecast features and writes an executable terminal headway plan. The lower
policy receives station-local high-frequency features plus compact same-time
APC/AVL context and chooses discrete holding. The current method also adds a
two-sided, pre-action departure-regularity reward to the lower learner.

The current controller does **not** enable the legacy holding guard, promotion,
leakage penalty, or DriftFB. Those modules remain historical development paths;
they are not part of the V8/V9 evaluated method and cannot be presented as
current contributions.

## Terminology ledger

| Term | Current paper meaning | Implementation anchor |
| --- | --- | --- |
| FreqDuet V6 | Frequency-separated asynchronous upper timetable and lower holding control under the locked physical and causal protocol. | `runner_v3.py`; `F_freqduet_protocol_v6_confirmed_main_hiro.yaml` |
| Causal harmonic prior | Ridge-fitted historical Fourier coefficients updated online by RLS after each completed APC bin. | `frequency/intensity_estimator.py` |
| Low-frequency demand | Harmonic rate, slope and 30-min forecast used by the upper policy, with high-frequency energy retained only as a scalar summary. | `DemandFrequencyTracker.upper_features()` |
| High-frequency demand | Pre-update station-local arrival innovation, innovation change, and local/global residual energy used by the lower policy. | `DemandFrequencyTracker.lower_features()` |
| Compact APC/AVL context | Load, capacity, queue, speed residual, shock age, schedule slack, and a causal two-sided holding target plus validity. | `lower/observation_contract.py` |
| Executable timetable | A 45-min terminal headway plan replanned every 15 min with rolling zero-sum phase conservation. | `upper/timetable_planner.py` |
| Lower action | One of seven holding times: 0, 5, 10, 15, 20, 30, or 45 s. | `lower.action_bins` |
| Two-sided regularity reward | Incremental change in predecessor/follower headway loss from evidence frozen before the sampled action. | `lower/causal_departure_regularity.py` |
| Protocol V6 reference | Same-semantics `F_freqduet_protocol_v6_noguard_hiro`; it retains the harmonic frequency path and also disables the legacy guard. | V8/V9 matrix manifests |
| Fixed headway | Strong external rule baseline under the exact V9 source/scenario contract. | V9 external comparison |
| Promotion/leakage/DriftFB | Disabled historical candidates, not current method components. | resolved config and Figure 1 method contract |

## Supported methodological statement

The defensible method statement is:

> FreqDuet V6 combines causal frequency-to-authority allocation with a compact
> APC/AVL lower state, an executable phase-conserving headway planner, discrete
> holding, and a two-sided incremental regularity objective.

The V8/V9 contrast evaluates this complete controller against the Protocol V6
reference. It is not an isolated holding-guard ablation: both arms disable the
legacy guard, and the current arm adds compact APC/AVL context together with
the regularity objective.

## Evidence statement

- V8 independently confirmed the registered 40-episode headway-CV effect with
  passenger-journey no-harm: CV delta `-0.02231`, 95% CI
  `[-0.03805,-0.00750]`; journey delta `-0.26266 min`, 95% CI
  `[-0.83661,+0.17494]`.
- V9 did not confirm the registered 200-episode regularity gate: journey delta
  `-1.24238 min`, 95% CI `[-2.20444,-0.53635]`; CV delta `-0.00911`,
  95% CI `[-0.02785,+0.00572]`.
- Against fixed headway under V9, FreqDuet lowered CV and restricted service
  cost but increased passenger journey by `+2.47025 min`, 95% CI
  `[+1.87431,+3.18799]`, and imposed substantially more holding and
  fleet-readiness denials.
- Against rule holding and rule MPC, FreqDuet lowered restricted passenger
  journey under the same V9 comparison.

## Claim boundary

The current confirmatory package does not contain same-stage NoFreq,
RawHistory, AllFreq, swapped-layer, promotion, or leakage controls. Earlier
engineering screens that include some of these arms are exploratory or tied to
superseded semantics. Therefore the current paper can report the behavior and
robustness of the complete controller, but it cannot claim that V8 alone
identifies the causal contribution of frequency separation.

The passenger endpoint is restricted total journey per generated passenger.
The V8 gate selected a regularity effect subject to journey no-harm; the
journey interval crossed zero. This must be described as a regularity result,
not a passenger-time improvement. V9 is a required negative robustness result
and may not be pooled with V8.

## External-data boundary

The public MTA AFC and Halifax APC subsets support a descriptive demand-shape
audit. MBTA boarding/alighting/load tables and MTA Bus Time route/AVL caches
support separate realism and calibration-readiness audits. They do not form a
same-day matched AFC/APC/AVL/OD calibration, a completed route-family policy
matrix, or a field-effect estimate.

## Paper positioning

The strongest current paper is a transparent methods-and-evidence account:

1. motivate exogenous frequency allocation as an HRL design question;
2. define the causal harmonic/APC observation contract and physical timetable
   execution contract;
3. report V8 confirmation and V9 failure together;
4. present fixed headway as a genuine passenger-time and fleet-readiness
   challenge rather than hiding it behind the service-cost scalar;
5. treat promotion, leakage, counterfactual value planners, and other failed
   variants as negative development evidence;
6. reserve a same-stage frequency ablation, matched route/day calibration, and
   learned first-stop/terminal value control for future confirmatory work.

The source-bound Methods, Results, tables, captions, and Supplementary Material
are assembled under `FreqDuet/paper/protocol_v6`.

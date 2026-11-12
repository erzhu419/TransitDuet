# Plan-Validity Paper: Remaining Evidence

Date: 2026-10-22. Computation is paused at the user's request. This document
records scientific gaps; it authorizes no scheduler submission or experiment.
The four prepared paths are integrated in the
[unified preregistration draft](../md/freq_hrl_unified_preregistration_2026-10-22.md).
It distinguishes the bounded in-distribution main claim from cross-stress
robustness. Two-stage freezes and two required comparators share a global
alpha0.05 allocation. Calibration costs, operational thresholds, historical
seed audit and confirmation artifacts remain pending; no formal run is authorized.

## Current conclusion

The paper now has one executable mainline: branch-supervised plan-validity
prediction followed by fixed-call-budget replanning. Stage9 provides a valid
independent-seed PointMaze result. The manuscript retains genuine learned-upper
and Transit single-intervention gains, but identifies their different methods
and units. Stage304's failed cost/wait, statistical wait-retention and critic-
increment criteria remain in the main Results.

This is a coherent bounded manuscript whose timing mechanism still needs stronger
comparative evidence within its intended multiscale planning/execution class.
Freq-HRL is not intended to replace general-purpose PPO/SAC learning backends.
The dominant issue is comparative and external evidence, not wording,
another encoder interface or more seeds on the same cached states.

## November 11 decision

Stop the boundary-conditioning upgrade. In the [frozen fresh-prefix test](../md/freq_hrl_boundary_conditioning_confirmation_2026-11-11.md#completed-result),
all 64 prefixes and 14,822,400 native steps completed without fitting. The
primary tracking gate is only 0.78% below Wide in MSE, with a crossing-zero
interval, a loss in the second prefix half and 93.43% of the net gain from one
prefix. Wide beats zero on three roots, but is 15.23% above zero on the primary
Reacher root at the point estimate. Keep the controls and all outcomes. Even
this prediction uses both realized boundary states at `t + 50`, unavailable at
decision time `t`. It neither validates a deployable trigger nor establishes
that useful continuation information is absent.

The [frozen episode diagnostic](../md/freq_hrl_local_episode_result_2026-11-10.md#complete-episodes)
is closer to the paper's endpoint: PointMaze is 4.83% below the selected
threshold and 14.28% below learned termination on two pilot roots, but the
second pooled path half loses to threshold. Reacher nearly ties threshold in
aggregate (+0.023% ISE); its two roots have opposite directions, and the one
that beats termination compares against a head that failed best-clock
competence. Better local prediction, wider coverage and separate calibration
have not closed this gap. The two-task C1/C2 protocol remains on HOLD.

**Next priority: a PointMaze-only strong-comparator study, not a critic search.**
Prepare a separate prospective protocol for the exact Stage9 renewal rule with
a shared controller, best constant clock, strongest development-selected simple
threshold and a competent learned termination. Keep optional clocks, inputs
available at a check and the 24-call contract comparable. Declare development
and fresh optimizer/evaluation roles, total cost caps and a practically relevant
episode ISE effect before evaluation. Include branch acquisition, fitting,
selection, scoring and p99 latency. Proceed to fresh roots only if both primary
comparators are eligible, the development effect is practically useful against
both, and budgets are measurable. This does not revise the two-task protocol.

Keep Reacher as a reported transfer boundary. Before any further Reacher critic
fit, require a distinct reason to expect decision value: on the existing
development bank, check whether adding continuation information to short-window
value improves *which checks* renew at matched renewal frequency, not whether it
lowers boundary MSE. Use disjoint window halves for selecting and scoring
actions, retain both directions and compare against the fixed short-window and
simple-timing choices. This is a finite-sample information diagnostic, not a
deployable policy or an upper bound. Without stable, material headroom, keep
this branch closed; with it, test causal prediction at `t` before another native
episode evaluation. The exposed readouts remain reporting-only. No new job is
released by this decision.

**PointMaze development gate passed.** The [separate Stage9 audit](../md/freq_hrl_stage9_strong_timing_2026-11-11.md#development-result)
reports 8.21% lower episode ISE versus the selected threshold and 23.45% versus
eligible learned termination, with both roots and path halves favorable and
all measured caps satisfied. This releases only the new PointMaze confirmation,
with its global freeze; it does not release the original two-task protocol.
The cached Reacher timing audit does not support reopening the tail critic.

**Fresh PointMaze builds: HOLD before evaluation.** All eight builds completed
and all method budgets passed. The frozen learned termination at root
1981440000 missed the best-clock competence requirement despite noncollapsed,
state-conditional behavior. No reserved test path was accessed and no root
was dropped. See the [build outcome](../md/freq_hrl_stage9_strong_timing_2026-11-11.md#build-outcome).
The [separate distance diagnostic](../results/pointmaze_stage9_distance_diagnostic_20261112/summary.json)
retains all eight frozen roots and supports a 7.28% ISE improvement on new
development paths, with all root means favorable. It does not replace the
withheld comparison with learned termination.

**Next comparator study.** Keep Stage9 and the threshold fixed. Before
accessing a new test cohort, register a strong comparator that chooses between
learned termination and the best clock on its own selection paths, and charge
both searches. Evaluate this composite comparator on every root and report
how often the learned policy is selected. Beating the composite would not
establish a learned-only claim where the clock is selected. Preserve the
original all-learned HOLD and leave its reserved paths untouched.

## Competent timing baselines and mechanism isolation

The strongest unresolved comparator is a learned termination/timing policy that
actually learns a useful noncollapsed behavior. Stage10's deadline collapse does
not establish superiority over Option-Critic, TEMPLE or a well-trained timing
head. Error/age/change-threshold heuristics are also plausible explanations for
the benefit. A valid algorithm comparison needs the same observations and branch
label budget, comparable capacity, and calibration selected without test outcomes.
The history-versus-current contrast currently changes 39 versus170 features;
it does not isolate history from feature construction or regularization.

The shared jitter-trained controller is appropriate for an executed timing
intervention, but not sufficient for ranking complete training algorithms.
Periodic and adaptive baselines need their own properly matched training
schedules, with inherited training/selection/label costs counted.

The [prepared baseline suite](../md/freq_hrl_strong_timing_baselines_protocol_2026-10-22.md)
now implements simple thresholds, equal-parameter/effective-DF ridge controls,
and paired learned termination with equal-capacity current/history heads.
It preserves collapsed best checkpoints and reserves test access until freeze.
The [completed native pilot](../md/freq_hrl_native_development_complete_2026-10-29.md)
and frozen episode diagnostic provide development results, not independent
strong-baseline confirmation or a faithful named-method reproduction.

## Same algorithm beyond one task family

The next generality evidence would have to keep the renewal target, fitting
recipe and timing contract fixed in a genuinely different control problem.
That problem should share the intended slow-plan/fast-execution structure;
universal superiority over underlying RL optimizers is not the target.
Transit currently changes binary keep/renew timing into finite-MC timetable
action search; its positive results therefore cannot fill this role. A new
evaluation would need independent training roots and full episodes, with task
and stress-family coverage separated from repeated futures of one state.

The [prepared second task](../md/freq_hrl_second_task_protocol_2026-10-22.md) is
moving-target Reacher on the standard articulated-arm dynamics. Both plants now
have a shared execution/label path and the same comparison/fitting function,
without changing the 50-step keep/renew target, ridge recipe or one-call/bin rule.
The native pilot and frozen Reacher episodes have since run. They do not show a
stable advantage over threshold on this second task.

Stage304 is a completed negative boundary, not a broken run to resubmit.
Its changed profile mix, callback positions and state distribution prevent
identifying repeated decisions alone as the cause. Any later Transit repair
would need a new development population and a retained no-repair comparator;
Stage298/301/304 outcomes remain confirmation/diagnostic evidence, not tuning data.

## Real compute and acquisition cost

Twenty-four upper calls is an exact count guarantee, not efficiency evidence.
The candidate additionally scores 84.1875 times per episode on average, and its
eight-root offline branch dataset replayed 1,942,606 primitive steps. Runtime,
latency, training/selection cost and branch-label acquisition must be compared
with competent baselines, alongside outcome-versus-budget curves.

The [prepared total-compute path](../md/freq_hrl_total_compute_budget_protocol_2026-10-22.md)
now measures complete single-method development pipelines on one-thread CPU
workers, including branch supervision, fitting, selection, scoring and all-step
control latency. Its budget selector charges every declared trial, including
failures, and admits configurations using development data only. Missing costs
block admission. Native pilot costs and latencies have been measured, but three
interrupted attempts still have unknown costs and no matched-lifecycle performance
result exists. The original fixed-call guarantee is unchanged.

Simulator event metadata selects the training queries and neutral windows.
Deployment features are causal and exclude event labels, but acquisition is
not fully unprivileged. Evidence with attainable query selection, or an explicit
simulator-training application scope, is needed for a broader deployment claim.

## From local value to episode outcomes

The existing proof gives exactly one call per bin. It gives no episode regret
bound or fleet-risk certificate for first-threshold-crossing decisions. A useful
analysis must account for the unobserved future opportunity cost of spending
the bin's call, proxy-horizon mismatch and policy-induced state shifts. A full
convergence theorem is not a substitute for competent baselines; a clearly
stated conditional proposition would be sufficient if its assumptions apply
to the implemented rule.

The [prepared reliability path](../md/freq_hrl_episode_reliability_protocol_2026-10-22.md)
now compares fixed-clock reference/deployed states, probes the unchanged local
target against adaptive full-episode continuations with independent conditional
futures, and separates opportunity and tail effects. Full-episode mean/tail ISE
and sustained-failure inference use frozen margins and independent paths/roots;
zero observed harmful events do not close the risk criterion automatically.
The frozen diagnostic found local and episode signs can disagree; the planned
independent-root full-episode risk claim has not been tested.

Practical effect size and robustness also remain central. Transit Stage301
service-only saves about 0.074s per single intervention; Stage304's service-only
point wait gain is 0.342s but total cost is worse at the point estimate and
rare fleet events dominate uncertainty. Larger task-relevant improvements and
broader decision coverage matter more than an arbitrarily small significant gain.

## Work possible during the compute pause

The active manuscript, terminology, source map and table artifact are rebuilt.
The old responsibility-restoration manuscript and figure contract are archived.
The three core figures, SVG/PDF/PNG exports, source CSVs and corresponding
manuscript captions are now produced from existing results; see
[figure evidence audit](figure_evidence_audit.md). A venue-specific submission
package remains unfinished. The related-work
update checks primary records for the current neighbors; it is not an exhaustive
priority search. No new result, universal claim or submission-readiness verdict
can be obtained merely by repackaging the existing positive stages.

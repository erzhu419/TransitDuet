# Plan-Validity Paper: Remaining Evidence

Date: 2026-10-22. Computation is paused at the user's request. This document
records scientific gaps; it authorizes no scheduler submission or experiment.

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
Preparation and unit tests do not close this scientific gap; no performance
run or faithful named-method reproduction has been completed for the new suite.

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
Only deterministic equivalence tests and a six-step Reacher interface check have
run; this is preparation, not evidence of second-task effectiveness.

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
block admission. No measurement or matched-budget performance result exists yet;
the original fixed-call guarantee is unchanged.

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
This is executable preparation, not a global guarantee or a new positive result.

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

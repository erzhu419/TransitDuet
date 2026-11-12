# Stage9 Strong Timing Comparison

[Frozen manifest](../protocols/plan_validity_stage9_strong_timing_2026-11-11.json).
PointMaze only: Stage9 history rule, strongest selected simple threshold,
learned history termination and best constant offset. Use one shared controller
per root, identical checks and 24 calls. The two complete paid pilot roots
supply the development weights/rules; no development refit. Compare actual
weights, reproduce original selection ISE and confirm native Stage9 parity.
Only then evaluate all four frozen methods on 32 new paired paths/root.

Release: two preflights (177,600 steps), then two audits (307,200 steps).
Admit measured pipelines under 10,000 CPU seconds at 100 deployments,
6.5 million offline steps and 10ms control p99. Charge inherited offline
supervision, training and selection; audit acquisition is separate.
Require competent termination at selection and audit, and at least 2% mean
episode ISE improvement against each comparator, positive in each root and
both pooled path halves. No replacement, recalibration or seed extension.

Eight fresh optimizer roots with 128 paired paths each are reserved, not
released until the development gate passes. This is an in-distribution timing
study under shared controller weights, not the original two-task C1/C2 study.
The [confirmation runner](../scripts/run_stage9_strong_timing_confirmation.py)
builds one controller per root and charges each comparator its share of the
measured development and full local build costs. All eight admitted builds must
freeze before evaluation: 4,096 paired-mode episodes, 4,915,200 evaluation
steps. Primary paired t intervals use eight optimizer roots; tail ISE and
failure counts at the existing 1.0/25-step threshold are descriptive.

## Development result

[Measured readout](../results/pointmaze_stage9_strong_timing_20261111/development_summary.json).
Native preflights t214802-t214803 reproduce controller tensors, selection ISE,
termination eligibility and original Stage9 calls/ISE/return. The two earlier
pairs failed at checkpoint comparison or stale launch staging, before native
steps; retain both. Audits t214804-t214805 complete 256 full episodes on 64
fresh paths. Total new PointMaze steps: 484,800, zero fits.

| Reference | Pooled ISE improvement | First / last half | Root 1 / root 2 |
| --- | ---: | ---: | ---: |
| Strongest threshold | 8.21% | 10.96% / 5.49% | 9.20% / 7.32% |
| Learned termination | 23.45% | 26.57% / 20.27% | 19.80% / 26.41% |
| Best clock | 39.11% | 42.94% / 35.02% | 40.18% / 38.12% |

Both learned termination rules pass the frozen selection/audit competence
criterion. All methods pass the caps (max charged 6,918 CPU seconds,
4,972,800 offline steps and 1.69ms p99). This passes the release gate, not
independent-training confirmation. The scheduler coordinator submits fresh
builds now and evaluation only after the global admission check; no waiter
jobs are queued. It retrieves only public ledgers and the fixed readout.

Launched `t214806-t214813`: all eight workers passed the development gate
and started one shared-controller build each on scheduler-selected
node003/004/006. [Release receipt](../results/pointmaze_stage9_strong_timing_20261111/confirmation_release.json).
The running coordinator will save the final decision in
`results/pointmaze_stage9_strong_timing_20261111/confirmation_decision.json`.

## Build outcome

All eight builds exited 0 and all 32 method budgets passed. The
[original decision](../results/pointmaze_stage9_strong_timing_20261111/confirmation_decision.json)
is nevertheless BUILD_GATE_HOLD: root 1981440000's learned termination is
noncollapsed and state conditional but trails its best fixed clock by
0.01718 selection ISE (1.37472 versus 1.35755). All original evaluation paths
remain unused. No root or comparator was replaced.

All eight selected thresholds use current tracking distance. Stage9 beats
these thresholds on five of eight selection means (3.87% pooled gain), but
selection paths are not independent evidence. The separate
[distance diagnostic](../protocols/plan_validity_stage9_distance_diagnostic_2026-11-12.json)
evaluates the same two frozen rules on 32 new development paths per root,
614,400 native steps, no fits. Root-level nominal 95% bounds, both path halves
and a 2% pooled improvement floor guide further development only; the original
confirmation decision and its reserved paths stay unchanged.

## Distance diagnostic result

[Result](../results/pointmaze_stage9_distance_diagnostic_20261112/summary.json).
t214816-t214823 completed 512 episodes on 256 paired paths at exactly 614,400
native steps and 559.85 CPU seconds, no fits. Mean ISE was 1.010906 versus
1.090286 (7.28% lower); paired root-mean advantage 0.07938, nominal 95% t
interval [0.03008, 0.12868]. All eight root means favor Stage9, ranging from
0.09% to 14.26%; the two path halves improve 7.14% and 7.42%. All 256 schedules
differ. Neither rule recorded a sustained failure at the existing threshold.
Peak root p99 latency was 1.73ms versus 1.25ms, with no 10ms deadline misses.
Only 17,986 final summary bytes returned; no checkpoints or traces.

This supports incremental value over current distance in the tested task and
controller setting. It does not isolate the effect of history from other
features, certify tail risk or establish superiority over learned termination.
The original HOLD remains.

## Reacher branch decision

The [cached readout](../results/pointmaze_stage9_strong_timing_20261111/reacher_tail_decisions.json)
uses eight original development prefixes and 24 C0-visited checks per root.
Rank W or G on two windows, score on the other two and reverse. Fix the
number of renewals at each clock to C0's count. Incremental G gain of G ranking
over W ranking: root1 -0.01478/0.00000, root2 +0.00791/+0.00115. Root1's
reverse ranking changes no check; both root2 intervals cross zero and its
G-versus-C0 gain changes sign. No stable evidence to reopen the critic branch.
These are finite-sample label-information contrasts, not deployable policies.
Zero fitting/simulation, only 5,341 derived summary bytes retrieved.

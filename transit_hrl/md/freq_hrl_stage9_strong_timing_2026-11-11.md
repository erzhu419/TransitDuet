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

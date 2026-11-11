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

# Stage216: frozen training fit versus domain transfer

Reuse Stage215 old/mean8 costs and Stage213 GLM/shrunk wait models.
Compare all600 training objectives, original240 training states, matched
same-root/regime scene-training references, and original240 held states.
Track cost pair MSE, wait mass excess, stable false-safe and feasible-cost
inversions in both future partitions. Compare scene/regime/root models on
exactly the same held states; report root directions and causal feature
range extrapolation. Preserve 240 weight8/360 weight1, independent of the
now uniform eight future draws. No model, threshold or budget changes.
Cross-fold cost comparisons use physical MSE, not fold-specific normalized
head MSE. Selected harm/benefit and stable selected-wait violations are separate.

One unbound node001-node006 scheduler task,1CPU/3072MB;94 cost/94 wait
models, zero fits/updates/episodes. Bank/targets/models/panels stay remote.
The initial r1 output is superseded by r2: remove noncomparable normalized
cross-fold MSE contrasts and add selected-wait harm/benefit decomposition.

Limits: training errors diagnose the current regularized fit, not feature
impossibility. Split training sizes differ; root also changes demand/prefix
seeds. Same-state contrasts are descriptive, not a pure lower-policy-shift
experiment or independent CI. Stability panels cover the original240 only.

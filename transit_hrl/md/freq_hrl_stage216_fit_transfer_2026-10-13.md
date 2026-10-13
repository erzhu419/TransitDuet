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

Completed r2 t151663, node006, exit0, source4147922982; all frozen
predictions/outcomes reproduced. Sixteen distinct tests passed; report1071524B.
Mean-cost/GLM scene matched-training stable inversions42.47/40.22%,
held44.60/42.38% (first-last/even-odd): fitting/order alignment already weak.
Shrunk root-split wait excess train0.23365->0.14492, held0.28412->0.30672:
extra fitting capacity worsens generalization. On identical root401 states,
cross-root stable false-unsafe58->81/150 and61->83/144, while false-safe
86->77 and88->76. Mean benefit0.011757->0.003432min; harm also falls
0.006624->0.005240min: the positive wait balance chiefly loses benefits.
Same-state wait-excess increments: regime+0.05629, root+0.00103 overall.

Next: fixed-feature train-only cost projection bound to distinguish the
linear-class fit floor from ridge/target alignment; wait needs benefit-side
false-unsafe recovery, not simply a stronger false-safe penalty or wider net.

Limits: training errors diagnose the current regularized fit, not feature
impossibility. Split training sizes differ; root also changes demand/prefix
seeds. Same-state contrasts are descriptive, not a pure lower-policy-shift
experiment or independent CI. Stability panels cover the original240 only.

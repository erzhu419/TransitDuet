# Stage214: bound reachability and frozen cost order

Stage213 mean8 training did not close cross-root transfer. Enumerate every
flexible wait mask under fixed +/-1 GLM-logit correction;original always
safe and frozen cost tie order preserved. Also enumerate unbounded masks.
Select the least wait-harm,then least physical-cost reachable action using
one future half,evaluate only on the other;reverse directions and repeat
even/odd. Mean8 hindsight is separate. Compare learned GLM/shrunk,wait-only,
cost-only,joint oracle,bounded/free reachable oracles. Count stable wrong
signs locked by the bound and joint opportunities blocked by cost order.

One scheduler task,unbound node001-node006,1CPU/3072MB;47 reused GLMs,
original240 states/40 groups/8 futures. No fits,rollouts or reserved scenes.
Raw/models remain remote;small report only.

Completed`t151478` on node004,exit0,source`7586f74a98`;23 tests passed.
One440803-byte report pull. Cross-root401 first/last:77 stable false-safe
candidates,0 locked by bound;0 joint-opportunity states blocked by bound,
16 blocked by cost order;100/216 stable feasible cost pairs inverted.
Wait-only oracle passes aggregate cost/wait across both partitions;
first/last joint-cost oracle root397 cross cost+0.002301969 despite
selection-half optimization. Cost ranking and future noise both matter;
increasing the residual bound is not the first repair.

Next: refit the same cost critic on complete mean8 targets,retain old
training scales/weights and frozen wait masks;reproduce the old-label
analytic control before interpreting the new-label comparison.

Limits: pointwise masks are optimistic relative to a shared network;
oracle half-selection is diagnostic,not a deployable policy or independent CI.

# Stage205: separate wait, cost ranking and action-space headroom

Freeze five Stage204 policies/47 folds; reuse240 states/40 prefixes/8 draws.
Draw0-3 select,4-7 evaluate, then reverse; even/odd is the fixed sensitivity.
LL=frozen,LO=replace wait,OL=replace cost,OO=replace both. Oracles select on
one half only; unconstrained cost and mean8 hindsight are separate controls.
Run: `native_transit_oracle_factorial_stage205_diagnostic_20261013_r1`.
Completed: `t147933`,node004,exit0; source `cb9320f661`;17 tests passed.
Unbound node001-node006,1CPU/3072MB; zero fits,updates or native episodes.
One manual report sync,833986 bytes; no draw,bank,model or checkpoint pulls.

Primary policy, scene split; both-direction mean changes relative to identity:

| Selector | First/last cost | First/last wait,min | Even/odd cost | Even/odd wait,min |
| --- | ---: | ---: | ---: | ---: |
| LL | -0.002193371 | -0.002673958 | -0.002193371 | -0.002673958 |
| LO | -0.000772779 | -0.008568750 | -0.001984157 | -0.008265625 |
| OL | -0.000431988 | -0.009238021 | -0.001623690 | -0.009050521 |
| OO | +0.000320329 | -0.015088542 | -0.001149562 | -0.015729167 |

Wait feasibility is the leading aggregate bottleneck: LO alone passes both
roots/all three split mean cost/wait gates in both partitions; LL and OL fail
wait. First/last scene has228 repeated false-safe candidate predictions and45
selected actions harmful in both halves. Cost ranking is also wrong:137/361
feasible pairs with repeated true ordering are inverted. But four-draw cost
oracle selection is noisy, not a superior ranker: OO same-half cost is
-0.010618437, cross-half +0.000320329, mean8 hindsight -0.008183955.
The menu is not globally empty:131/240 mean8 joint opportunities,79/240 with
one jointly beneficial action in both first/last halves (85/240 even/odd).
26 states have mean8 cost opportunity without joint feasibility;83 lack even
mean8 cost opportunity. These are sampled local limits, not impossibility.

Next: freeze cost and the menu. Change the learned wait training target from
pair-MSE shrinkage tuning to draw-level feasibility with a false-safe-sensitive
loss, retaining grouped training/validation and causal test-time inputs.
Check high-noise cost and burst wait separately before expanding actions.

Limits: LO is a label-informed diagnostic, not a learned policy. Its first/last
reverse-direction root397 cost is positive in all splits; even/odd passes all
directions, so cost robustness is unresolved. LO saves less cost than LL.
High-noise scene cost and burst scene wait remain harmful even under LO;
the gate averages regimes within each root, not safety in every regime.
Halves share their physical prefix and latent demand profile;40
prefix groups remain40, not80 or480. Replacing predictions with sampled
labels diagnoses mechanisms; it is not a deployable policy or independent
CI, and cannot establish impossibility if the sampled frontier fails.

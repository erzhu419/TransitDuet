# Stage155: Delayed Upper Credit

Stage154 rejected learned adaptation beyond constant departure phase. Test one
specific learning bottleneck before replacing the upper action representation:
service credit closes every ~180 s, while dispatch affects subsequent route service.

Add eight-decision soft Retrace versus unchanged one-step RE-SAC. Store behavior
density at collection, use `c=lambda*min(1,pi/mu)`, and propagate soft Bellman
residuals with terminal/truncation masks. Lambda=0.9; generic trace core lives in
`freq_hrl/core/retrace.py`. This follows the return-correction construction of
[Munos et al. (2016)](https://arxiv.org/abs/1606.02647), with sampled soft next values.

Four cells: two methods x fresh roots347/359. Keep physical-L1 unit critic,
service reward, gamma, bounds, lower policy, frequency routing and network sizes.
300 episodes, 2,700 upper/9,000 lower updates per cell; trace uses more target-Q
evaluations, not more collected data or optimizer steps. Qualification is separate
from fresh full training. Evaluate learned/zero-upper/fixed7/zero-holding on twenty
paired scenes, last checkpoint only. Scheduler node001-006, no local native run.
69 focused tests pass, including terminal/truncation masks, ratio indexing,
delayed-credit propagation and bit-exact preserved one-step optimization.
Stage153's archived summary is unchanged after the shared worker extension.

Run `native_transit_trace_credit_stage155_development_20261010_r1`: t141036-t141039
are DONE. All four cells pass pairing, qualification, frozen-network and update
budget checks: 1,200 training episodes, 320 frozen episodes, 93,297,600 native
ticks plus 129,600 independent qualification ticks.

Retrace actually propagates: effective trace mass 4.074/3.611, mean absolute
target correction 0.512/0.504, coefficient 0.834/0.779, no target clipping.
Roots347/359 learned-minus-own-fixed7 results:

| Learner | Cost delta | Restricted wait delta (min) | Reward delta |
| --- | --- | --- | --- |
| one_step | -0.021802 / -0.020348 | -0.00905 / +0.00280 | -0.473 / -1.580 |
| retrace8 | -0.022407 / +0.018525 | -0.01315 / -0.02295 | +1.602 / +0.228 |

Retrace-minus-one-step cost is -0.010359/+0.040607, reward +47.960/-4.341.
Own-neutral cost is worse for both Retrace roots (+0.000143/+0.018880).
Retrace's fixed7 cost contrasts contain fleet-step components -0.020833/+0.020833;
other components sum to -0.001574/-0.002309. Positive pooled wait/reward contrasts
therefore do not establish a robust service-cost gain or state adaptation.

Judge both whole-training contrasts and own learned-minus-fixed7/zero-upper;
record effective trace mass and actual target correction, not just horizon=8.
Decision: stop L1/trace/seed expansion on the scalar phase interface. Next use
endpoint-anchored service-interval allocation, retaining the trip budget and
communicating the executable interval plan to the lower. Qualify physical plan
authority before paying for another full native training matrix.

## Limitations

Two-root mechanism test, not a claim of general HRL. Multi-step credit does not
repair the limited per-trip phase action or establish a neural convergence proof.
Retrace's original guarantees are not asserted for this soft neural implementation.

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

Judge both whole-training contrasts and own learned-minus-fixed7/zero-upper;
record effective trace mass and actual target correction, not just horizon=8.
If credit propagates without useful adaptation, redesign upper plan variables.

## Limitations

Two-root mechanism test, not a claim of general HRL. Multi-step credit does not
repair the limited per-trip phase action or establish a neural convergence proof.
Retrace's original guarantees are not asserted for this soft neural implementation.

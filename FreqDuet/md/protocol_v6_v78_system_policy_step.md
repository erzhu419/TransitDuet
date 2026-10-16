# V78: constrained complete-day policy-step pilot

Stop the V72-V77 action-value selector family. Start from the original V34
B200 lower actors for seeds 12011/933013, not V59's irregular post-64 actors.
Freeze those backbones, upper/B200 deployment, all critics and production duals.
Keep seven holding bins and their physical execution unchanged. This stage tests
constraint learning before changing physical holding permissions.

Learn four nonnegative logit-tilt coefficients from zero: causal two-sided
departure-regularity gain, negative APC load times holding, negative published
overdue-trip pressure at the return terminal times holding, and negative
cumulative holding on the current trip times holding/target headway. Normalize
action by 45 s. Follower evidence is current AVL, not a realized future arrival;
missing bilateral evidence contributes zero to that feature. No learned Q,
counterfactual action label, marginal forecast or latent queue enters the tilt.
Zero tilt reproduces B200's categorical law; sampling and execution use the same
tilted distribution, with V69 logical-event uniforms for matched evaluation.

Collect 32 independent training days 1780001-1780032 per original seed under
the unchanged stochastic policy. Use once-only full-day passenger return-to-go
(units 1,000 passenger-hours); terminal arrival CV-squared and mean readiness
delay are separate constraints. Score gradients are action features minus their
conditional policy expectation. Baselines use other independent demand days
only: time-aligned remaining passenger costs and terminal constraint means.
Do not subtract the same-day CRN reference inside a policy-gradient baseline.

One four-dimensional SLSQP step per seed minimizes the observed passenger
directional cost, subject to nonpositive CV/readiness directional costs, exact
training mean state KL <= .001 and each coefficient in [0,1]. No fitted critic,
actor epoch loop, multiplier tuning or reward-weight sweep. The constraints
are local Monte Carlo estimates, not a guarantee of closed-loop feasibility.
Freeze both coefficient vectors before reading any held-out rollout. If no
feasible improving step is identified, retain zero and mark no update; reuse
identical references rather than rerunning/claiming strict improvement.

Held-out days 1781001-1781016: two original seeds, deterministic mode once
and stochastic mode with three logical-event sampler repeats. Each cell pairs
zero tilt and frozen candidate from day start: 128 pairs/at most 256 rollouts,
plus 64 fitting rollouts. Primary is matched stochastic deployment; deterministic
results are separate, not a learning-versus-sampling contrast. Average sampler
repeats/seeds within demand day for paired intervals. Joint gate: passenger CI
upper <0; CV-squared, readiness, unlaunched and incomplete CI uppers <=0. Main
admission additionally needs the deterministic gate, further seeds and long-run
confirmation; this pilot alone never replaces it.

Server smoke: six tests, exact zero-tilt historical action/episode replay, and
one nonzero-tilt full-day collection with conserved exposure. Then scheduler CPU
shards collect training days, fit both adapters, run held-out pairs and aggregate
on node001-node006. Keep trajectories/checkpoints server-side; pull tiny markers
and final JSON only. No local experiments or default-environment package changes.

Next: judge actual closed-loop outcomes, gradient variability and concentration.
An unchanged or failed step does not justify another value selector or long
training. Only a repeatable joint improvement merits a separately specified
multi-step learner/physical-permission study. Limitations: one local step, only
two original training seeds, noisy episodic gradients, approximate AVL feature;
stochastic feasibility does not establish deterministic feasibility or convergence.

## Execution

Source `c742fa1894`; smoke `t166129` passed six tests, exact historical B200
episode/holding identity and a nonzero full-day collection. Prototype integration
failures (JSON integer type, runtime-only interface assumptions, native action
quantization accounting) were fixed before formal collection; obsolete queued
tasks were cancelled. Four 16-worker training shards `t166131-t166134` completed
all 64 training-day rollouts, exit 0 with no retries. Fits `t166136/t166137`
froze nonzero coefficients for both original seeds before held-out evaluation:

- Seed 12011: approximately `(0, .127912, 0, .251250)`.
- Seed 933013: approximately `(0, .357135, 0, 0)`.

Both observed mean state KL values are .001; estimated passenger and readiness
directional costs decrease while estimated CV-squared costs do not increase.
These are training estimates, not measured closed-loop improvements. Neither
fit activates the two-sided-gain or return-backlog coefficient materially.

Held-out tasks `t166142-t166145` cover all 128 pairs in one wave on
node002/004/005/006, 32 workers each. Server aggregation `t166146` waits for all
four small shard markers and applies the unchanged joint gates separately by mode.
Main remains frozen; no held-out outcome is claimed at dispatch.

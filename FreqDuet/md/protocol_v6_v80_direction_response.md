# V80: paired complete-day policy-direction response

V79 finds unresolved regularity gradients in both V78 fits and all four
half-training transfers. Mean summed conditional KL is 5.2 per day; normalized
path-weight effective days are 1.20/2.75 out of 32. These are distinct warnings:
the data do not identify whether noise or finite-step nonlinearity dominates.

Freeze the original B200 backbones and both V78 directions. On 32 fresh demand
days 1800001-1800032, two original seeds and two logical-event sampler repeats,
run the same complete day at scales 0, -.1, +.1 and 1 of its frozen direction.
Use V69 matched uniforms. Negative tilt is a diagnostic perturbation, not
negative holding: seven nonnegative physical action bins remain unchanged.
128 independent cells / 512 rollouts, stochastic deployment only. Reuse the
first full smoke cell in the matrix. All run through scheduler on node001-node006.

Save the zero-tilt score data server-side. Replicate the on-policy score
direction with other-demand-day baselines within each sampler repeat. Compare
it against the paired centered response (J(+.1)-J(-.1))/.2, the positive-side
response, and the original full-scale effect. Contrast full-scale versus
centered response to diagnose the local approximation. Use demand-day units,
averaging repeats and actors; recompute score baselines in the day jackknife.
Report paired uncertainty for score minus centered response, not independent
error bars treated as a paired test. Metrics: passenger person-seconds,
CV-squared and readiness delay. The centered response is a finite difference,
not an exact derivative or a safety guarantee.

This is estimation diagnosis, not a step-size sweep for promotion. Do not
choose a scale, fit a controller, change V78's gate, or use these development
days as later confirmation. If the centered response identifies a constraint
cost while the score remains noisy, test a paired-response estimator next.
If the local and full responses differ materially, redesign update locality
before control learning. If neither resolves, report that rather than launch
long training. Actor, critic, dual and main remain frozen; pull only small
markers and the final JSON, not trajectories or checkpoints.

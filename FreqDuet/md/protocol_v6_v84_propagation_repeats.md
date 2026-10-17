# V84: repeat fixed-prefix action propagation

V83 has one future per prefix and concentrated, mixed-sign feedback effects.
Repeat all 24 registered prefixes, without dropping seed 933013/day 1810009.
Keep B200 actors, V78 coefficients, V81 projection, dual and main policy fixed.

Each prefix receives eight new post-decision futures, seven physical holding
bins and both continuation modes: 192 fresh native reference trajectories and
2,688 labelled arms. Native arms reuse their reference; other arms are complete
day replays. No fitting, gate tuning, training or controller promotion.

Fork only after matching the complete V83 pre-action observation. Keep sampled
pending arrivals, current speeds and day-level demand/OD/peak latents unchanged.
Reseed later scenario innovations, runtime streams, upper/lower sampling and
the actual logical-event categorical sampler. Match future seeds across all
seven actions and both modes, independently across futures. Each future gets
its own recorded-lower tape; the old V83 tape is not used after its fork.
The frozen upper actor continues reacting to physical state in both modes.

Smoke checks unchanged original-future endpoints and whole event/command path,
exact prefix identity, distinct future uniforms, fresh-native identity in both
modes for both actors, and a changed-action recorded tape with zero later lower
command changes. Tests, references, arm shards and aggregation run exclusively
through scheduler on node001-node006. Pull only small markers/summaries.

Report weighted raw/projected-minus-B200 single-decision effects, paired mode
differences, seven-action response and paired versus independent-future noise.
Intervals use eight futures conditional on each prefix, not actions as samples
or training-seed inference. Compare all original V83 effects, including its
concentrated adverse case. Missing downstream events remain missing. These are
consumed-prefix mechanism diagnostics, not fresh-day confirmation or additive
attribution of the complete deployed policy effect. Decide on system credit or
control changes only after examining repeatability and noise.

Source `bd0f737cc7`; smoke `t166295` passed all nine unit tests and complete
prefix/native/tape checks for both actors. Reference task `t166296` uses 48
workers. Shards `t166298-t166301` cover `[0,669)`, `[669,1338)`, `[1338,2006)`,
`[2006,2688)` on node002/004/005/006 with 48/48/48/49 workers. Aggregate
`t166302` waits for these exact four markers. Only the 70-byte smoke marker was
pulled manually at dispatch; no scientific or promotion claim was made then.

All rollout shards completed exit 0/retry 0. Initial aggregation failed only on
NumPy integer JSON serialization; source `7daaf32f5d`, task `t166313` passed the
new regression and rebuilt the complete summary without rerunning rollouts.
Only compact logs were read. V83 closed-loop CV-squared stratum means
`+9.43e-6/+1.00e-5/+5.14e-5` become `-6.21e-7/-5.47e-7/+6.08e-7`.
The concentrated active/inactive passenger effects shrink from +1324/+4727 to
+15.9/+82.7 person-s, both conditional intervals crossing zero. Residual harms
remain in individual prefixes; this is not evidence of global non-harm.

Server audit in `t166316` finds only 8/144 passenger and 7/144 CV action contrasts
with absolute mean above paired future SD in closed loop, versus 69/144 and
73/144 with recorded lower. Closed-loop paired/independent SD medians are
.0342/.0849. Feedback contrast intervals have six positive and four negative
prefixes for each primary cost. These are descriptive consumed-prefix counts,
not independent action samples or multiplicity-adjusted confirmation. No
controller promotion. V85 tests a single delayed-feedback mechanism contrast.

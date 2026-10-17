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

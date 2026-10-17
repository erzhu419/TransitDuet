# V83: single-action holding propagation

V82 identifies missing bilateral evidence and conditional gap bias, but completed
gap replay is not an alternative-action label. Local departure gains can coexist
with worse full-day arrival CV. Separate the immediate holding response from
feedback through later lower decisions before designing another controller.

Keep B200 actors, V78 coefficients, upper deployment state and all seven physical
holding bins fixed. Reuse the four consumed V82 days and stochastic repeat 0:
eight reference rollouts. In each, register the first decision at/after 7200 s
in each causal stratum: missing bilateral, projection active, valid inactive.
Selection sees only the current observation and projected distribution, never
future outcomes. These 24 contexts each receive all seven actions under two
continuations: closed-loop projected policy, or reference lower commands keyed
by logical trip/stop/direction after the intervention. The upper actor still
reacts to the physical state in both arms. The recorded lower tape is a diagnostic
control, not a deployable policy or an oracle training label.

There are 336 labelled arms; native-action arms reuse their exact reference.
Record actual adjacent departure gaps, next/third/fifth-stop arrival gap losses
where available, follower ready/departure/holding, and complete-day passenger,
CV-squared, readiness and service-completion outcomes. Preserve simulator event
order for same-tick ties. Match pre-action state, commands and arrival prefix;
the native-action smoke must reproduce the whole original command/arrival path.
A changed-action tape smoke checks that later lower commands remain fixed.

For each registered distribution, sum its probability difference from B200
against all seven action outcomes. Compare the same conditional expected effect
under closed-loop versus recorded-lower continuation. Positive predicted gain
means improvement; negative measured cost difference means improvement. This
isolates lower-continuation sensitivity but does not additively attribute the
whole V81 policy effect. One fixed future per prefix and four consumed days
support mechanism diagnosis, not generalization, a new gate or promotion.

Tests, reference collection, independent action arms and aggregation all use
scheduler on node001-node006. Pull only small summaries/markers. No actor,
critic, dual or fitted-value update; no V81 threshold tuning or main switch.

Source `d4c0c354f7`; smoke `t166281` completed exit 0/retry 0. Five tests pass,
the reference exactly reproduces its V81 endpoints, both forced-native modes
reproduce the entire command/arrival path, and a changed-action recorded-lower
replay preserves every later lower command. These smoke arms are reused.
Reference task `t166282` collects the remaining seven references using eight
workers. Shards `t166283-t166286` cover `[0,84)`, `[84,168)`, `[168,252)`,
`[252,336)` on node002/004/005/006 with 84 workers each, gated by reference
completion. Aggregate `t166287` waits for their four completion markers.
All results remain diagnostic; the main policy is unchanged.

All references, 336 labelled arms and aggregation completed exit 0/retry 0.
Only the 79,928-byte result JSON was pulled. Recorded-lower subsequent action
changes are exactly zero. Projected-versus-B200 single-decision conditional
CV-squared effects (closed-loop / recorded-lower) are: missing +9.43e-6/-4.20e-4;
active +1.00e-5/-6.53e-6; inactive +5.14e-5/+7.13e-4. Immediate departure and
next-stop loss means agree between continuation modes, but full-day effects
differ. Lower feedback can amplify or suppress effects; it is not uniformly
harmful. Closed-loop cost increments are concentrated in seed 933013/day 1810009.
One fixed future per context cannot establish repeatability or policy advantage.
Next: repeat identical prefixes with independent post-prefix innovations and
matched policy uniforms, retaining all three strata and both continuation modes.
Establish repeatable action signal before a system-credit/control update. No
threshold tuning, blanket lower freeze or policy promotion follows these means.

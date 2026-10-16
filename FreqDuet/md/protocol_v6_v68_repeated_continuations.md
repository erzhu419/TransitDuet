# V68: repeated matched continuation audit

Freeze V59 actors, upper/B200 deployment, dual, value models and main. Reuse all
96 V67 development prefixes, not only its 38 hindsight winners. Eight fresh
continuations per prefix, seven actions each: 768 groups / 5,376 formal rollouts.
Native action runs first without override; six alternatives share its future.

Replay the original V67 history, draw the current actor action, and match the
full V67 prefix before changing any random state. Future tape seed is
`68000000 + context_index*100 + future_index`; role-separated private lower/
upper, global Python/NumPy/Torch and runtime NumPy streams restart only then.
Preserve current speed, scheduled current-window passengers and the realized
day-level demand/OD/rush profile. Subsequent arrival windows and speed updates
use fresh innovations. All seven arms must match post-fork RNG state exactly.

Smoke: tests, one no-fork full-day V67 identity, two complete future/action
grids and forced-native full holding/upper/day identity in both. Seventeen
rollouts, fourteen reused formally. Only scheduler CPU nodes node001-node006;
raw groups/CSV remain remote, local pulls are small JSON/markers/logs.

Primary diagnostic: preserve V67's original hindsight action, including its
grid tie rule, and evaluate it on all eight new continuations against the
native action in each matched future. Secondary: select using futures 0-3,
evaluate on 4-7, then reverse and average. Selection minimizes mean passenger
exposure subject to mean CV-squared/readiness/unlaunched/incomplete no worse
than native; new selection ties preserve native. Report held-out passenger and
constraint deltas, not the in-sample oracle as controller benefit.

Also report action-effect noise variance and mean-effect squared corrected by
variance/8 (keep negative estimates), half-future sign agreement excluding
double-zero ties, and frozen-critic differences against eight-future mean
labels versus zero-difference prediction. Average actors/windows/futures
within each of the eight original demand-day clusters before bootstrap.

Limitations: conditional on exact simulator prefix and fixed realized day
profile, including already sampled current-window arrivals; not integration
over a posterior of latent day profiles given observations. These are consumed
development prefixes from two original training seeds, not fresh-prefix or
controller confirmation. No fitting or policy promotion in V68. Use the
repeatability/noise result to decide whether action-increment learning is
supported and what uncertainty it must represent; fresh days remain necessary.

## Execution

Source `ea40a3b82f`, snapshot `source/v68_repeated_20261016`.
Smoke `t165496` passed: 53 tests, V67 no-fork identity, two complete future grids
and both forced-native identities. Matrix `t165498-t165502`: 257 planned workers
over three waves on node001/002/004/005/006. Aggregate `t165503` depends on all
five completion markers. Only the small smoke JSON was pulled locally.

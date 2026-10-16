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

## Outcome And Decision

All 5,376 formal rollouts and aggregate `t165503` completed with rc=0/retry=0;
all prefixes and post-fork RNG starts matched. Passenger deltas below are
full-day person-seconds averaged over the 96 prefixes, negative is better.
V67 selection on eight new futures: +22.38, CI [-2630.13,1961.00]. Four-future
selection evaluated on the other four: +1892.41, CI [290.95,3541.78]; CV-squared
also worsened by +0.00011960, CI [0.00005424,0.00018739]. Same-eight-future
oracle gain -6922.16 is an in-sample selection result, not replicated benefit.

Passenger debiased signal-squared was -8.71e6, CI [-6.73e7,3.77e7]; CV-squared
signal was 1.54e-7, CI [-1.16e-8,3.34e-7], versus noise variance 6.17e-6.
Half-future sign agreement was 47.95%/47.40%. No critic reliably beat zero-delta
prediction: best shrunk-linear MAE 0.00065434 versus zero 0.00062858.

Posthoc audit `t165618` (source `25e574b72a`, no added rollouts): among V67's
38 strict winners, 18 improved and 20 worsened on new-future means. Their
all-prefix contribution was +0.54, CI [-2570.47,1934.71]; three zero-gain ties
do not explain the missing benefit. No fixed single-action probe confirmed
passenger improvement; 45s worsened it. Mid-window CV has exploratory weak
signal, about 2.6% of single-future noise variance, without passenger support.
The failed inline audit `t165617` exceeded SSH argument length and was canceled;
the script-based replacement completed. Only compact JSON was pulled locally.

Decision: do not fit an action-increment controller from these noisy labels.
Next use a small development-only coupling comparison: align future policy
random draws by logical trip/stop event instead of global decision order,
while preserving categorical probabilities, actor weights and the action set.
Check whether this reduces paired-label variance before expanding futures or
fitting a scorer. Decision-order coupling is a hypothesis, not an established
cause. Main, actor, dual and value remain unchanged.

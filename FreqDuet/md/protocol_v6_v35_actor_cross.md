# V35: Execution-Actor Cross at a Fixed 200-Episode Context

V34 found rising holding and passenger journey time in all three policies despite
improving absolute headway CV. This diagnostic crosses upper/lower actor states
from episodes 39 and 199 (40 and 200 completed training episodes), separately
within each V34 configuration and training seed. All other loaded state and the
evaluation policy episode remain at 199. Only `policy_net.state_dict()` is
replaced; critics, temperatures, multipliers, sampling streams, belief, objective
weights, cached selectors, configuration, and action semantics stay fixed.
This fixes initial deployment state, not the endogenous within-episode evolution
of queues, demand filtering, holding feedback, and executable plans.
Frozen inference uses the online upper actor; the EMA upper is training-only.

The design is 3 configurations x 12 training seeds x 4 combinations x 12 shared
scenario seeds: 144 independent CPU jobs and 1,728 frozen rollouts, with no
training. The verified optimized V9 snapshot is used on node001-node006 through
scheduleurm. Models and raw results stay on the server; only summaries return.
Smoke tests reproduce unchanged 200/200 evaluations against V34 and exercise an
early-lower mixed actor; non-actor state is checked before/after each transplant.

Write Y(u,l) for an outcome at this common background, with 0=40 and 1=200.
Report lower changes at both upper ages, upper changes at both lower ages,
their averaged effects, and interaction Y11-Y10-Y01+Y00. Averaged upper and lower
effects sum to Y11-Y00; interaction is not added again. Subtracting the original
40-episode V34 outcome from Y00 gives a separate residual for non-actor context
and evaluation stage, not a pure belief-state effect. Both components sum to
the observed 40-to-200 change after unchanged-control reproduction succeeds.

Primary outcomes are restricted journey time, cumulative holding per launched
trip, headway CV, and fleet readiness delay. Use the original crossed-bootstrap
intervals and V34 cohorts. APC holding penalties, regularity reward, and upper
wait/backlog credit are descriptive mechanism outputs. This is reused-seed
attribution, not an independent confirmation or selection of a new main policy.

Regression `t152598` passed 16 tests. Smokes `t152599-t152601` reproduced 390
unchanged-control columns in each configuration and passed mixed-actor frozen
evaluation. Diagnostic source `263ded1317` is frozen server-side. Evaluations
`t152635-t152778` are running, 36 each on node001/node004/node005/node006;
aggregation `t152779` awaits their small completion markers.

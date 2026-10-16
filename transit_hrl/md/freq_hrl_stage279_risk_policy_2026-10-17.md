# Stage279: frozen risk-head action utility

- Run: `native_transit_risk_policy_stage279_20261017_r1`.
- Motivation: Stage278 observed-state matched-df risk improves pooled prediction errors over both controls; test whether the same frozen predictions improve decisions.
- No fitting. Read the seven cached clipped/raw risk arms from Stage278; reuse the same raw reference, service contribution, wait logits, candidates and native selector.
- Score: raw total + (arm risk - raw risk). This preserves raw decisions exactly and changes only the fleet-risk contribution; original-first ties and feasibility are unchanged.
- Reuse all 32 cells, 704 model contexts and 2,048 queries; full train/held, prefix retained/excluded and same prefix heads on outer held.
- Freeze all actions per cell before reading candidate outcomes. Evaluate assigned/opposite16 and fixed 8+8 halves.
- Report paired total-cost, fleet, service and wait deltas to raw; comparisons to Linear9 and matched-df action context use identical populations. Primary: mixed-pooled excluded/opposite and full60-held/opposite.
- Policy denominators include every query state, including states with no eligible nonoriginal action. Risk-MSE replay retains its existing eligible-state weighting. Repeated exposure is not a seed.
- Budget: one unpinned scheduler CPU task, 3072 MiB, any node001-node006; zero new fits, simulations or actor updates. Cached model/state decision exposures: 84,480.
- Large actions and outcome ledgers stay remote; pull scalar group report only.
- Verification: five focused tests cover exact raw decisions, original-first ties, feasibility, physical-cost and half identities, zero-fit cell/group replay, action freezing before any outcome access, and the scheduler parent gate; both cold CLI startups passed.
- Completed: `t166113`, node006, done/0, no retries. Raw risk, risk-error sums and group risk MSE reproduce exactly; cap excess 1.04e-17. All 84,480 repeated model/state exposures evaluated with zero fits/simulations. Pulled only 1,375,168-byte scalar report.

## Results
Mixed-pooled opposite16 outcomes relative to frozen raw choices; negative is better. Cost columns x 1e6; wait is minutes.

| Population | Linear9 cost | Observed-df cost | Linear9 wait | Observed-df wait |
| --- | ---: | ---: | ---: | ---: |
| Full60 train | +157.907 | +23.195 | -0.000286882 | +0.000151790 |
| Prefix54 retained | +86.662 | -20.697 | -0.000319922 | +0.000175799 |
| Prefix54 excluded | +172.546 | +173.864 | -0.000167546 | +0.000206608 |
| Full60 outer held | -414.689 | -366.298 | -0.001265495 | -0.000706380 |
| Same prefix54 heads, outer held | -432.667 | -387.227 | -0.001117204 | -0.000899281 |

Observed-df versus Linear9: excluded fleet -32.552e-6 + service +33.870e-6 = cost +1.318e-6; held fleet -27.127e-6 + service +75.517e-6 = cost +48.390e-6. Wait worsens +0.000374154/+0.000559115 min; both future halves agree on service/wait harm. Cost contrast changes sign between halves in both populations.
Observed-df versus matched-df action context: excluded cost +140.518e-6 (worse), held -276.974e-6 (better). Thus Stage278 MSE gains do not establish policy superiority. Observed-df changes 238/1,920 excluded and 203/1,920 held decisions relative to raw; 1,920 is repeated exposure, not independent seeds.
Next Stage280: fixed-action predicted-versus-actual service/risk margin attribution; compare both critics on identical disagreement pairs, count fleet events and preserve 8+8 halves. No new fitting, choice changes or simulation.
- Limitations: cached single-intervention plug-in evaluation on reused development futures; not full-episode HRL, new-prefix confirmation or training-seed CI. No selector/default change. Stage253/248/169 remain closed.

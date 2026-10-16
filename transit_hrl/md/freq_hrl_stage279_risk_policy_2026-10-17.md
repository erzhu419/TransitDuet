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
- Status: source ready; scheduler submission pending.
- Limitations: cached single-intervention plug-in evaluation on reused development futures; not full-episode HRL, new-prefix confirmation or training-seed CI. No selector/default change. Stage253/248/169 remain closed.

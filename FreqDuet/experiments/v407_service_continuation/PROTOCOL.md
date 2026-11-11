# V407: Continuous Service Inside the Rollout Window

Freeze the corrected V406 single-action predictor, two demand scenarios, cost
weights, twelve physical vehicles and seven legal bins. No paired actions,
release edits, trained values or main-policy change. Original sources stay intact.

The new kernel dispatches every already-published trip due inside its existing
lookahead, instead of stopping after 360 seconds per vehicle. Known public demand
and evaluation clocks bound arrivals and simulation. At the window boundary,
active vehicles and passengers remain explicit, and unlaunched due trips accrue
censored public/readiness delay. Unfinished trips are not marked complete.

Single-action selection keeps cost/CV2/delay nonharm against the same always-active
10-second reference. Finite-window accounting additionally prevents abandoning
a trip the reference dispatched or leaving more total passenger stock. These
checks are forecast constraints, not a guarantee about real all-day outcomes.

First run synthetic same-input native-rule, service-continuity, end-clock, stock
balance and residual-obligation checks; measure old/new evaluation workload.
After engineering acceptance, at most sixteen new development paths: two nominal
days and two heavy days x fixed10/fixed15/frozen-V406/corrected-single. No traffic
retry or test-day parameter selection. Both simple rules apply from the first
query and use the same public timetable. Fixed15 is an operating-point comparison,
not an exact holding-dose match or a proof about the whole tradeoff frontier.

Primary screen: mean journey improves within each regime relative to fixed10,
with actual CV2/public-delay/readiness nonharm on every paired day. Report the
frozen-V406 correction contrast and fixed15 comparison separately. Keep the V406
negative decision unchanged; no automatic confirmation or policy promotion.
Everything runs scheduler high/manual on node001--006, local JSON summaries only.

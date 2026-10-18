# V111: consumed-path action and disturbance accounting

V110 is closed: all three service/regularity/readiness gates failed. Do not
rescue its coefficient or enlarge its sample. Main, actor, critic and dual stay
frozen. This stage reads the existing 72 V110 pair/NPZ records on a server through
scheduler, with no simulation, checkpoint loading, training or new demand day.

Reconcile every positive executed holding bin against saved action counts;
reconcile summed executed seconds per launched trip and actual arrival CV2
against V110 endpoints. Initial unobserved zero-hold events are reported
separately. Report the seven commanded-bin counts, zero/45s fractions, executed
holding dose, and the already-saved unrounded request saturation. Average
samplers within actor/day before the existing crossed actor/day intervals.

At arrivals, describe adjacent-stop persistence of nonholding elapsed contrasts
for the same predecessor trip, including prior own holding and the next actual
hold. This is arrival-state accounting, not a reconstruction of ready-time
states. The saved NPZ does not contain follower estimates or feasible masks.

Test the control law's response to balanced current forward/follower gaps with
nonzero cumulative R. At D=0, R=-60s still requests 45s while the original rule
requests zero. Repeated use of a persistent level is therefore not automatically
an incremental disturbance correction. This analytic example establishes a
property of the law, not how often it caused the measured service loss.

The next decision is whether existing records suffice to quantify repeated
compensation. If exact decision evidence is needed, first instrument consumed
V110 paths and reproduce their endpoints; no new gain or fresh-day controller
screen precedes that diagnosis. A structural successor must account for already
executed compensation/current gap response rather than blindly reapply a
cumulative level. Only summary/status.json returns locally; details stay remote.

## Outcome And Next Step

Frozen source `3c47037de5`, scheduler `t183650`, node006. Four tests passed and
all 72 paths reproduced positive executed-bin counts, holding s/trip and actual
CV2. No accounting discrepancy was found. Retrieved only the 26,784-byte JSON;
no new rollout or checkpoint load occurred.

Compared with bilateral025, FF raises actual 45s command fraction from 1.933%
to 29.361%, zero-command fraction from 46.007% to 50.538%, and average command
from 7.770s to 17.110s. Native/preserve cap fractions are 15.269%/13.972%.
The candidate is more polarized, not simply active at more stops. Its larger
dose produces worse passenger time, CV2 and readiness, so it supplies no
evidence of useful compensation.

Same-predecessor negative arrival residuals persist after own prior holding in
14,932 candidate adjacent-stop cases, with 491,890s current-stop holding, versus
16,828 cases/192,875s for bilateral025. These unequal, policy-dependent arrival
supports do not identify repeated ready-time compensation or mediate the loss.
The balanced-gap control-law test confirms the level term can still request
45s after the current gap is balanced. Exact ready/follower evidence is missing.

Next: passive decision traces on consumed V110 paths, reproducing the cached
endpoints first. Record matched predecessor identity, current forward/follower
gaps, release/holding/nonholding coordinates, prior executed compensation,
feasible bins, unrounded request and chosen command. Quantify repeated level
response and saturation before any structural successor or new-day screen.
No gain refit, actor/critic/dual update or main-policy switch is authorized by
these accounting results.

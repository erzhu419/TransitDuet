# V251: earlier causal linkage and compensation tradeoffs

Design/synthetic seal only. No archive scan,shadow request,replay,fit or training;
V238 `b4b1d00203`,main,fleet and seven-bin permissions remain unchanged.

Reuse V232 current-plan/ready-FIFO/incoming-position association. Start a WATCH
at the first observed unbacked future link;it is not known future shortage.
V238 already uses future links:an earlier trigger alone is not a new method.
Store first query clock,physical cycle,plan/version,rank/distance,posted lead,H.
Keep own future and other due exposure separate. No ETA or new predictor.

An episode requires the same physical cycle and currently associated plan ID.
Plan revisions/future-to-due changes preserve its start;loss/reassociation ends
it immediately. A regained plan starts a new episode. Physical H never resets
at these changes. Historical links never become current reservations or slack.
The reference is disconnected from runtime and outputs no action.

Future read-only estimands:all-cycle first-link vs first-own-due timing/H;
within-episode same-plan lead;association churn and first-next same-bus labels.
Keep lost/never-due/no-query/missing cases. Do not choose successful episodes
retrospectively or attribute all intervening H to the current obligation.

Compensation remains an independent axis using frozen PREPARE evidence:
retain all feasible a with a-h and R(a)-R(h)=2(a-h)[a+h-(B-F)]. H is not deducted
from F/B again. Positive local regret is an unresolved system tradeoff,not a
reason for a forced cut or a strict-local-nonharm guard. No weighted score,
uniform attenuation,larger gain,threshold tuning or online bin mapping.

Later causal tests retain V246 controls,independent selection/evaluation
futures,actual passenger/CV2/readiness/public-delay/completion gates and the
resource-specific due-obligation criterion. Early watch cannot replace them.
Next gate:separately authorize one read-only audit of the two accepted V249
paths. No selector or new path is admitted by this seal.

## Limitations
Current position order does not identify return order or assignment. Future
lack of current ready stock does not prove shortage. First-query observation
cannot reconstruct earlier unseen state. Synthetic correctness establishes
neither archive support nor control benefit.

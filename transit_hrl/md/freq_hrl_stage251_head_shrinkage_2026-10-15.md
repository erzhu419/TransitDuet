# Stage251: Training-Only Head-Difference Shrinkage

Stage250 free heads improve some decisions but rank16 fits training noise and
transfers poorly. Keep the old total RMS cap; test one training-only difference
regularizer against frozen Stage250 frozen_old/free_old and Stage247 old controls.

Let S be the service+risk correction, D its difference, and D0 the cached
difference. H is the difference Hessian of the state-balanced paired GLS loss.
Estimate Vb, covariance of its difference score, from the assigned training
futures, preserving both action and service/risk covariance. State innovations
are independent; covariance of each state's sample mean is summed over states.
V = H_pinv Vb H_pinv holds fitted precision, reliability and S fixed.

Obtain the auxiliary free fit under the old cap, with increment d = Dfree-D0.
Signal energy is max(||d||^2-trace(V), 0). If positive, lambda = trace(H V)/signal;
otherwise freeze D=D0. Refit GLS + lambda/2 * ||D-D0||^2, subject to the same old
sum RMS cap. Zero estimated noise gives the free endpoint when signal is positive.
This regularizes learned coefficients in addition to the existing target shrinkage.

No penalty/rank grid, evaluation calibration, cross-arm winner selection or
scenario gate. Same features, TSVD, 8 inner/16 full training futures, opposite16
evaluation, A2 OOF selector, four views and two bank directions. Read all three
control models/selections/actions verbatim. Auxiliary free solves estimate the
new penalty; they do not replace or update the frozen control.

Report selected and all four alpha1-rank training-bank/seen-prefix new-future/
held-prefix new-future errors, feasible-pair and nonanchor head-difference MSE,
cost/wait outcomes, admission, noise/signal energies and coefficient changes.

Budget: 32 fits + one report, 5,376 auxiliary free and 5,376 shrunk solves, zero
native episodes/raw-reference fits/actor updates/new prefixes. One successful
preflight before the remaining 31. Scheduler node001-node006, one core/task,
unbound. Pull only scalar receipts and compact report; reuse server caches.

## Outcome And Next

Registered 32 fits + report completed done/0: preflight t162315, remaining
t162319-t162349, report t162355. Post-hoc cached diagnostic t162369 also done/0,
without additional fits or episodes; its rationale is in `diagnostic_addendum.json`.
All 53,760 frozen action exposures reproduce exactly; 24 focused tests passed.
Only the 1,871,882-byte report, 535,386-byte diagnostic and tiny receipts were pulled.

Rank16 mean standardized difference-increment RMS falls 0.734110 -> 0.196586
(73.22%); 16/32 mixed full fits freeze the difference. Selected fleet nonanchor
MSE improves 2.08% versus free, but feasible-pair MSE worsens 0.67%. The rule is
active and reduces large head corrections, but does not improve control.

Mixed held-prefix cost/wait deltas versus raw, negative is better:

| Model / Selected Keys | Cost | Wait / min |
|---|---:|---:|
| Frozen difference | -0.000064318 | +0.000030924 |
| Free / Free | -0.000126306 | -0.000281217 |
| Shrunk / Shrunk | -0.000065757 | +0.000015951 |
| Shrunk / Free (diagnostic only) | -0.000117465 | -0.000197103 |

Changing model first, then selected keys, attributes 85.40% of the pooled cost
regression and 71.69% of wait regression to key changes. Bank0 accounts for the
regression; bank1 key changes slightly help. This path-dependent decomposition
does not establish that the selector is always wrong. All 17 fixed candidates
are diagnostic, not a new ranking/selection grid; no model/default is promoted.
Stage247 frozen old still has better pooled cost and wait point means.

Next: freeze models and candidate grid. Compare old A2 OOF validation labels,
matched new2 and expanded new8, with validation prefixes excluded from the inner
fit and new draws taken from the other half of its training bank. Never select
on the opposite16 evaluation bank. This separates old-future sensitivity from
validation information count; do not increase the difference penalty again.

Limitations: V is a conditional plug-in estimate, not full joint-fit uncertainty,
tail-risk certification or a CI; variance weights/reliability and the constrained
sum are estimated too. Stage249 futures and prefixes are development data.
Stage248 is not read; Stage169 reserves stay closed. No default/deployment change.

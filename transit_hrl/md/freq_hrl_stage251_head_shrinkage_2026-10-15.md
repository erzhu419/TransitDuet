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

Limitations: V is a conditional plug-in estimate, not full joint-fit uncertainty,
tail-risk certification or a CI; variance weights/reliability and the constrained
sum are estimated too. Stage249 futures and prefixes are development data.
Stage248 is not read; Stage169 reserves stay closed. No default/deployment change.

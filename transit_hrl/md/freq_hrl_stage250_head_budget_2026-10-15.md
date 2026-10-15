# Stage250: Fixed-Data Head/Budget Factorial

Stage249 reduced target-mean error 64.30%, but selected held-prefix fleet MSE
improved only 0.13%. Test two restrictions separately using its existing cache.

| Arm | Head difference | Total correction RMS cap |
|---|---|---|
| frozen_old | Cached, fixed | Cached A radius; Stage249 expanded control |
| free_old | Learned | Cached A radius |
| frozen_train | Cached, fixed | Training-calibrated radius |
| free_train | Learned | Training-calibrated radius |

Training-calibrated radius is max(cached A radius, norm of the OLS projection of
the shrunk training service+risk residual onto the frozen whitened TSVD basis).
It uses only the assigned 8 inner/16 full training futures. No multiplier grid.
Free difference is analytically profiled; the same convex sum-ball solver remains.
Total-cost function dimension is k in all arms; freeing heads gives 2k component
degrees of freedom rather than k. Unsupported difference directions retain cache.

Freeze references, six actions, features, ranks, scales, state-GLS loss, variance
and shrinkage formulas, A2 OOF selector, and suffix. Select each arm independently
with the old rule; never choose an arm from Stage249/250 evaluation outcomes.
Reuse frozen Stage249 expanded and Stage247 controls without refitting them.

Report both bank directions and four views. Compare training-bank, seen-prefix
new-future and held-prefix new-future service/fleet/total MSE, all four alpha1 rank
curves, feasible-pair errors, admission, correction budgets and cost/wait outcomes.
Head contrasts: free_old-frozen_old and free_train-frozen_train. Budget contrasts:
frozen_train-frozen_old and free_train-free_old. Retain the factorial interaction.

Budget: 32 fits, one report, 16,128 new quadratic solves; zero native episodes,
raw-reference fits, actor updates or new prefixes. One end-to-end fit preflight
must pass before remaining 31. Scheduler node001-node006, one core/task, unbound.
Pull only receipts and compact report; models/panels/actions stay on the server.

## Outcome

All 33 tasks done/0: preflight t162258, remaining fits t162264-t162294, report
t162295. All 38,400 frozen control action exposures reproduce exactly. Twenty-two
tests passed. Only the 2,319,659-byte report and small scalar receipts were pulled.

Mixed-pooled held-prefix policy deltas versus raw (negative is better):

| Arm | Cost | Wait / min |
|---|---:|---:|
| Stage247 frozen old | -0.000227372 | -0.000766732 |
| frozen_old / Stage249 expanded | -0.000064318 | +0.000030924 |
| free_old | -0.000126306 | -0.000281217 |
| frozen_train | -0.000082327 | -0.000132617 |
| free_train | -0.000100512 | -0.000673275 |

Free_old improves cost/wait point means versus frozen_old in both banks, but
bank1 wait versus raw remains positive. Free_train bank1 cost is positive. Neither
arm surpasses Stage247 on both pooled outcomes. No winner/default is promoted.

Selected free_old fleet nonanchor MSE worsens 1.48%, while feasible-pair MSE
improves 0.61%. At fixed rank16, feasible-pair error improves 8.80% on training-bank
seen prefixes, only 0.59% on new futures at those prefixes, and worsens 3.48% on
held prefixes. Nonanchor errors worsen even on seen prefixes, not just transfer.
Head freedom helps some decisions but also exposes fitting noise and extrapolation.

Training calibration expands only 40/128 mixed full-fit caps, including 4/32 at
rank16. Frozen_train improves fleet nonanchor MSE 0.48% but worsens pair MSE 0.36%.
This is not evidence about unrestricted caps. Maximum sum/difference KKT < 2e-12.

Next: a fixed-cache contrast of cached, fully free and training-shrunk head
differences under the old total RMS cap. Control variance/cancelling head errors
before new-prefix confirmation; do not simply add futures or relax total caps.
Receipts, exact derived quantities and constraint counts are in this run's
`completion_receipt.json`, `findings.json` and `constraint_summary.json`.

Limitations: existing development prefixes and reused Stage249 futures, not fresh
confirmation or independent seed CI. Stage248 is not read; Stage169 reserves stay
closed. Independent-head benefit is not presumed; cap relaxation may worsen transfer.
No deployment/default change. New-prefix confirmation follows a separately frozen
decision after this factorial, rather than tuning on its held-prefix outcomes.

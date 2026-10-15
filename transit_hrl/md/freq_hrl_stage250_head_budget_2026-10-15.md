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

Limitations: existing development prefixes and reused Stage249 futures, not fresh
confirmation or independent seed CI. Stage248 is not read; Stage169 reserves stay
closed. Independent-head benefit is not presumed; cap relaxation may worsen transfer.
No deployment/default change. New-prefix confirmation follows a separately frozen
decision after this factorial, rather than tuning on its held-prefix outcomes.

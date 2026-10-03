# Stage103 Preflight Result

t130042/t130043 both finished on node001 with exit0. Server-only source-cell requalification passed, and the saved official summary equals the read-only reaggregation exactly.

- All12 donor loads/freezes,12 compositions, both matched training budgets, paired noise seeds and original source/Adam freezes passed. Stage103 performed48 native evaluations /14,400 steps; no learning, replay or checkpoint writes. Native evaluation wall18.57s.
- One root, four paths per policy/period, horizon300: joint-conditioned minus staged-common is -0.089760 at50 and +0.092496 at100. These are descriptive preflight effects, without CI or performance admission; retain both signs and all18 contrasts in the compact summary.
- Proceed with the already frozen eight-root full evaluation: six policies, both periods,32 fresh paired paths/policy/period, horizon1200. Total3072 episodes /3,686,400 steps,96 server-only donor loads, zero new checkpoints. Both corrected primary CI lower bounds must exceed zero; no donor/seed/period selection or tuning.

## Scope
This compares matched method-path training recipes with different registered training rosters, not isolated update-order causality or equal historical campaign compute. It does not reopen frequency-superiority or Stage67 critic claims. Only compact JSON and completion metadata were retrieved.

Full dispatch: t130084-t130091 are the eight root evaluations; t130092 is the dependent qualifier. All were queued at the 2026-10-03 09:31:56 UTC snapshot, eligible for dynamic node001-006 placement. Evaluation tasks request9 CPU /8 GB each; qualifier1 CPU /2 GB. Source code0283d61d99, preregistration08b7ca2ee0 and preflight evidencefe202c680c are fixed. Full performance was pending at dispatch.

Full result: all nine tasks finished with exit0; the two-period joint-superiority claim was not supported. See [the full result](freq_hrl_stage103_joint_staged_result_2026-10-03.md) and the archived compact summary.

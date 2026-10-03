# Stage104 Paired-Order Preflight Result

t130151/t130152 finished with exit0 on node005/node004. All source-cell qualification checks passed, and read-only reaggregation exactly matches the official summary.

- Four learners used matched actor-specific scenario/noise rosters. Joint collected both actors' credit before updating; staged completed lower then upper, retaining both phase freezes. All32 mean updates,24 update operations,8 phase-boundary checks and cumulative nominal KL checks passed.
- Exact cost:304 native episodes /91,200 steps,144 extra upper replay forwards, zero checkpoint writes. Native wall83.86s; compact pull27,609 bytes, no raw evaluation rows or checkpoints.
- One root, horizon300, two updates per actor, four evaluation paths/policy/period: joint-conditioned minus staged-common is +0.000499 at50 and -0.000377 at100. These descriptive effects have no CI or performance admission; all18 contrasts and both signs remain archived.

Next: unchanged eight-root full training, both periods and all four learners from original Stage96 teachers and Stage97 decoder, with shared actor rosters. Budget68,608 episodes /82,329,600 steps,1024 mean updates and64 final server-only checkpoints. Both primary corrected CI lower bounds must exceed zero.

## Scope
Mechanical eligibility does not establish joint superiority. Stage103's not-supported result stays separate; this remains same-task MC mean learning, not full actor-critic or frequency superiority. Stage67 critic HOLD remains unchanged.

Full dispatch: t130162-t130169 were the eight root training tasks; t130170 was the dependent qualifier. At the 2026-10-03 11:39:13 UTC snapshot, training was running and qualification queued. Code03ec49cce9, preregistrationb0c68644fa and preflight evidencedfae80d45c stayed fixed. All nine tasks subsequently completed with exit0. The [full result](freq_hrl_stage104_paired_order_result_2026-10-03.md) retains the not-supported two-period joint-superiority claim; conditioning benefits are positive under both update orders.

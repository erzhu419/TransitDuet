# Stage103 Joint Versus Staged: Full Result

t130084-t130092 all finished with exit0. All eight roots passed server-side qualification; source-cell reaggregation and a separate equal-root bootstrap calculation exactly reproduce the saved summary.

| Joint-conditioned minus staged-common | Mean reward difference | Bonferroni18 corrected CI | Positive roots |
| --- | ---: | --- | ---: |
| period50 | +0.317438 | [0.021487,0.677563] | 7/8 |
| period100 | +0.129327 | [-0.845860,1.109587] | 5/8 |

The preregistered two-period joint-superiority claim is **not supported**. All18 contrasts contain12 positive,2 negative and4 inconclusive results. Both recipes improve over the original teacher at both periods. Staged conditioning remains positive: +0.466147 CI[0.201052,0.778868] at50; +2.233131 CI[1.061550,3.536215] at100. Joint conditioning is inconclusive at50 (+0.455373 CI[-0.060829,1.025310]) and positive at100 (+1.700957 CI[0.776422,2.824274]). Base-minus-zero is negative at both periods.

Exact cost:3072 native evaluations /3,686,400 steps,96 donor loads/freezes,16 matched-budget checks, zero training or checkpoint writes. Native wall84.58-98.26s/root. The42,094-byte compact pull retains all endpoints and per-root means/effects; raw evaluation rows and checkpoints remain server-only.

Follow-up: [Stage104](freq_hrl_stage104_paired_order_result_2026-10-03.md) implemented and completed paired training with identical actor-specific scenario/noise rosters, initialization, decoder and update budgets. Its two-period joint-superiority claim is also not supported; it remains a separate cohort.

## Scope
The current recipes used different training rosters, so this is not an isolated update-order effect. The evidence is same-task teacher-initialized MC mean learning, not full actor-critic or frequency superiority. Stage102 is retained as a separate cohort; its positive at50 conditioning CI is not pooled with this inconclusive replication. Stage67 critic HOLD remains unchanged.

# Stage104 Paired-Order: Full Result

t130162-t130170 all finished with exit0. All eight roots passed read-only server qualification; reaggregation and an independent equal-root bootstrap exactly reproduce the official summary.

| Joint-conditioned minus staged-common | Mean reward difference | Bonferroni18 corrected CI | Positive roots |
| --- | ---: | --- | ---: |
| period50 | +0.016219 | [0.007150,0.023579] | 7/8 |
| period100 | +0.004062 | [-0.041508,0.037348] | 6/8 |

The preregistered two-period joint-superiority claim is **not supported**. With shared actor-specific training rosters, the observed update-order effects are small relative to learning gains; this does not establish equivalence. All18 contrasts contain13 positive and5 inconclusive results.

Lower conditioning is positive in both orders and periods, with8/8 positive roots for all four contrasts: joint +0.720295 CI[0.420747,1.153095] at50 and +1.917287 CI[0.853495,2.776622] at100; staged +0.713097 CI[0.410429,1.143925] at50 and +1.903039 CI[0.857505,2.765819] at100. All four learners improve over the original teacher at both periods. Base-minus-zero is inconclusive at both periods, so this cohort does not establish a positive source-actor residual effect. The registered zero variant retains the ridge forecast reference; it is not a flat/no-planning baseline.

Exact cost:68,608 native episodes /82,329,600 steps,1024 mean updates,768 update operations,64 phase-boundary checks,147,456 extra upper replay forwards and64 final server-only checkpoints. Native wall2071.35-2153.65s/root. The101,842-byte compact pull contains all contrasts and per-root means, effects, freeze/KL summaries and counters; raw evaluation, trajectories and checkpoints remain server-only.

Next: test the conditioning benefit under a separately registered task/initialization, rather than tune update order or expand this cohort to chase its small difference. No new protocol or task was dispatched in this result turn.

## Scope
This is same-task Stage96 teacher-initialized MC actor-mean learning with the Stage97 decoder, std, values, Adam and forecaster fixed. It is not full actor-critic, cross-task generalization or frequency superiority. Realized trajectory KL was not matched; common-noise training adds replay work relative to independent credit. Stage102/103 remain separate cohorts, and Stage67 critic HOLD is unchanged.

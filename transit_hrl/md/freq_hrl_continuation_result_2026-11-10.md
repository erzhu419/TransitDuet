# Prefix And Continuation Result

All eight shards t214267-t214274 exited 0: 32 new paths, 96 paired clocks,
eight conditional futures/clock, 6,976 native rollouts and 8,371,200 steps.
Main cost was 7,950.417 CPU seconds; no fits or optimizer updates occurred.
Only public aggregates were retrieved. No subsequent experiment jobs were submitted.

## Complete Episodes

ISE is lower-is-better; each task retains both frozen roots and all 16 paths.

| Task | History | Checks/clock | Relative change |
| --- | ---: | ---: | ---: |
| PointMaze | 1.012554 | 1.028831 | +1.61% |
| Reacher | 4.325697 | 4.335857 | +0.23% |

Each task has opposite root-level directions. Three roots reverse direction
between their four-path episode halves; the first PointMaze root worsens in both.

## Diagnostic Findings

At fixed candidate prefixes, replacing R continuation by C raises PointMaze's
noise-corrected W MSE to 2.63x/2.61x. Both future halves agree in direction.
Keeping R continuation while changing prefixes does not raise its mean error.
Reacher is different: changing prefixes under R gives 0.26x/11.24x error on its
two roots. Path1972240001 supplies 85.9%/84.6% of root2's prefix-error increase
under R/C continuation, yet its episode ISE improves by 0.842279. No path is removed.

A post-hoc cache-only A-to-B/B-to-A sign-selection readout separates information
from the frozen score/cut. Positive action gains are improvements. W-sign
selection improves local gain over C on all
four C/C root means. On Reacher root1 the W gain difference is +0.001644
[+0.000340,+0.002948], but G gain changes -0.004490 [-0.015180,+0.006199], with the
same tradeoff in both directions. Zeroing the frozen score's cut helps PointMaze
root1 but hurts PointMaze root2 and Reacher root1. G-sign does not consistently
dominate W-sign. Neither a universal threshold fix nor a direct switch to G is justified.

## Next And Limits

Next isolate label continuation: fix candidate-visited TRAINING states, features,
DF8 and calibration procedure; compare R/C continuation labels with shared
training futures. First test one-step improvement under the fixed continuation,
then complete episodes. Keep this round's 32 revealed paths out of calibration.
No new experiment has been launched.

Limits: common clocks cover 47.6%-81.1% of the clock union. Error-contrast path
intervals cross zero despite future-half agreement. Two roots/task and unadjusted
conditional path CIs do not rank population causes. Four-future sign selection
is an information diagnostic, not a deployable policy or an oracle. Termination
competence, unknown old lifecycle costs and the original HOLD remain unchanged.

Sources: [summary](../results/plan_validity_continuation_20261110/summary.json),
[audit](../results/plan_validity_continuation_20261110/execution_audit.json),
[cross-future readout](../results/plan_validity_continuation_20261110/decision_readout/public/result.json).

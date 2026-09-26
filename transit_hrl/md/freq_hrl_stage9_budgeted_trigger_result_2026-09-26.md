# Stage-9 Budgeted Trigger Development Result

Date: 2026-09-26

Run: `pointmaze_budgeted_trigger_stage9_v1_development_20260926_r1`

All eight frozen optimizer-root tasks (`t100586`-`t100593`) completed. The
registered analyzer accepted all eight compact `result.json` files and returned
`stage9_development_gate_passed`. The statistical unit is the optimizer root;
each root has 16 held-out trigger-evaluation paths.

| Paired contrast (candidate advantage) | Root mean [95% CI] |
|---|---:|
| Fixed minus candidate episode tracking ISE (primary) | 0.396372 [0.275784, 0.516959] |
| Random-offset minus candidate episode tracking ISE | 0.538632 [0.461288, 0.615975] |
| Current-only minus candidate episode tracking ISE | 0.215568 [0.116089, 0.315047] |
| Candidate minus fixed episode return | 30.770657 [21.111415, 40.429899] |
| Untrained minus trained controller tracking ISE | 15.889297 [14.339155, 17.439439] |

The independent calculation from held-out episode rows reproduced every
interval above. All eight roots have positive candidate-minus-fixed ISE and
return contrasts. Mean episode ISE was 1.474969 (fixed), 1.617229
(random-offset), 1.294166 (current-only), and 1.078597 (candidate). All four
modes made exactly 24 upper calls per 1200-step episode. The candidate made
12.796875 pre-deadline calls per episode on average, so the trigger was active.
The 320 path seeds across training, checkpoint selection, branch fit, and
trigger evaluation were disjoint.

**Claim boundary:** The frozen PointMaze development gate passed. It supports
a fixed-budget closed-loop improvement on this hidden-regime task, not an
independent replication, a cross-domain result, or a general Freq-HRL claim.
The next test must freeze the same algorithm and analysis on fresh optimizer
and path seeds before viewing their outcomes.

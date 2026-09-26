# Stage-9 Independent-Seed Confirmation Result

Date: 2026-09-26

Run: `pointmaze_budgeted_trigger_stage9_v2_confirmation_20260926_r1`

Tasks `t100612`-`t100619` completed on node001/node004/node005/node006.
All eight frozen optimizer roots and 128 held-out trigger-evaluation paths
(512 paired-mode rollouts) passed the registered result contract. The
confirmation set has 320 distinct
train/selection/branch-fit/evaluation path seeds, with no overlap with Stage-9
development. No optimizer roots were appended.

| Registered contrast (candidate advantage) | Root mean [95% CI] |
|---|---:|
| Fixed minus candidate episode tracking ISE (primary) | 0.505541 [0.408443, 0.602640] |
| Random-offset minus candidate episode tracking ISE | 0.608186 [0.570111, 0.646260] |
| Current-only minus candidate episode tracking ISE | 0.260773 [0.152216, 0.369331] |
| Candidate minus fixed episode return | 40.662456 [31.548406, 49.776506] |
| Untrained minus trained controller tracking ISE | 20.862987 [15.939428, 25.786547] |

All five preregistered lower bounds are positive: **the independent-seed
PointMaze confirmation gate passed**. A separate calculation from episode rows
reproduced the intervals. All eight roots had positive primary ISE and return
contrasts. Mean ISE was 1.447072 (fixed), 1.549716 (random-offset), 1.202303
(current-only), and 0.941530 (candidate). Every mode made exactly 24 upper
calls per episode; the candidate made 13.195312 pre-deadline calls on average.

**Claim boundary:** This confirms fixed-budget, plan-validity-guided closed-loop
improvement on fresh seeds from the same PointMaze hidden-regime task family.
It does not establish a frequency-band attribution, cross-task transfer, or a
domain-general Freq-HRL result. Development and confirmation are reported
separately; neither substitutes for transfer validation.

# Stage91: fixed final upper did not establish a lower repair

t122943-t122951 all done/exit0; mechanical gate passed. Eight roots,12,288 episodes /14,745,600 steps; wall364-403s/root, peak4507-4745MiB. Pulled76,077-byte compact JSON only; checkpoints stay server-side.
LC = lower trained with final Stage88 UJ frozen from update1; LM = Stage90 source-upper lower; LJ = Stage88 joint lower. All26 endpoints use the frozen equal-root bootstrap65536 / Bonferroni26 family on fresh evaluation seeds.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| Primary, fixed UJ: LC minus LJ | -0.00537 [-0.01479, +0.00427] | -0.01207 [-0.03557, +0.01417] |
| Fixed UJ: LC minus LM | -0.01385 [-0.03274, +0.00616] | -0.03074 [-0.05957, +0.00693] |
| Residual, fixed U0: LJ minus LM | -0.00847 [-0.01993, +0.00292] | -0.02068 [-0.03637, -0.00468] |
| Direct upper, fixed LJ: UJ minus U0 | +0.24409 [+0.16950, +0.32915] | +0.51949 [+0.41134, +0.65406] |
| Direct upper, fixed LM: UJ minus U0 | +0.24409 [+0.16957, +0.32872] | +0.51748 [+0.41042, +0.65223] |
| Direct upper, fixed LC: UJ minus U0 | +0.24428 [+0.16963, +0.32946] | +0.52188 [+0.41352, +0.65549] |

LC-LJ and LC-LM are inconclusive at both fixed uppers and periods. Only3/8 roots improve LC-LJ at either period: the fixed-final-upper repair is unsupported. Direct upper gains remain positive at all8 roots for all three lowers and both periods, with all six corrected CIs positive.
LJ-LM remains negative in mean; its supported period changes from50 in Stage90 to100 in Stage91. Source-minus-zero is negative at100 and inconclusive at50.
All26 CIs, costs, evaluation-mean contrasts and additive identities were independently recomputed. Each update used64 episodes;128 lower updates,16 initialization checks,32 donor loads,16 final checkpoints and128 compositions match the frozen budget. Upper/std/values/Adam freeze passed.
Next: stop this repair branch without adopting LC or retuning seeds/radius. A further lower intervention needs a distinct preregistered mechanism; keep the supported upper contribution separate from the unresolved conditional-lower effect.

## Limitations
Same teachers and paired training, so the residual comparison is fresh evaluation rather than independent training replication. Final UJ changes both upper level and temporal path; this does not isolate nonstationarity. Crossing-zero CIs establish neither equivalence nor harm. This is fixed-std MC mean learning, not full actor-critic or equal-total-compute comparison; Stage67 HOLD and the closed frequency-superiority claim remain unchanged.

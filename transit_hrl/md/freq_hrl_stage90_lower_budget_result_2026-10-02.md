# Stage90: budget reservation explains only part of the lower deficit

t121945-t121953 all done/exit0; eight roots,12,288 native episodes /14,745,600 steps; mechanical gate passed. Wall395-410s/root, peak4494-4672MiB. Only69,452-byte compact JSON pulled.
LM alone received128 lower-mean updates on the exact Stage88 training rosters;16 new final checkpoints remain server-side. Read32 Stage88 donors,128 exact actor compositions; no upper/critic/forecaster/Adam updates or raw traces.
U0 = source upper, UJ = Stage88 learned upper; LJ = joint lower, LL = full-budget lower-only, LM = new matched-budget lower. All28 endpoints use the frozen equal-root bootstrap65536 / Bonferroni28 family on fresh Stage90 evaluation seeds.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| Budget, fixed U0: U0/LM minus U0/LL | -0.01647 [-0.01888, -0.01184] | -0.01160 [-0.01516, -0.00810] |
| Matched-budget training, fixed U0: U0/LJ minus U0/LM | -0.01282 [-0.02138, -0.00451] | -0.01624 [-0.03254, 0.00353] |
| Budget, fixed UJ: UJ/LM minus UJ/LL | -0.01574 [-0.01854, -0.01123] | -0.01149 [-0.01475, -0.00806] |
| Matched-budget training, fixed UJ: UJ/LJ minus UJ/LM | -0.01252 [-0.02091, -0.00443] | -0.01376 [-0.03050, 0.00687] |
| Total lower gap, fixed U0: U0/LJ minus U0/LL | -0.02928 [-0.03874, -0.02188] | -0.02784 [-0.04276, -0.00871] |
| Total lower gap, fixed UJ: UJ/LJ minus UJ/LL | -0.02826 [-0.03788, -0.02106] | -0.02525 [-0.04041, -0.00518] |
| Direct upper: UJ/LJ minus U0/LJ | +0.26545 [0.18706, 0.35813] | +0.50214 [0.39090, 0.61743] |
| Upper with matched lower: UJ/LM minus U0/LM | +0.26515 [0.18692, 0.35766] | +0.49966 [0.38901, 0.61487] |
| Joint minus lower-only: UJ/LJ minus U0/LL | +0.23616 [0.16507, 0.32072] | +0.47430 [0.36996, 0.57604] |

Budget components are negative at all8 roots, both periods and both fixed uppers. At U0, their descriptive point-estimate shares of the mean deficit are56%/42%. The matched-budget training residual is supported negative at period50; period100 remains inconclusive, not equivalent to zero. Budget alone cannot explain the period50 deficit.
Every update used all64 episodes. Actual lower KL/update was .00098967-.00099045 at50 and .00099419-.00099587 at100, with nominal cumulative .00792/.00796; max old-logp replay error3.05e-5. Direct-upper and net joint gains remain positive at all8 roots and CI-supported.
Next: preregister lower training with the final Stage88 learned upper frozen from update1, same lower budget and training rosters, then fresh evaluation against LM/LJ; no upper retraining, budget search or checkpoint adoption.

## Limitations
Same teachers and intentionally paired Stage88 training, not an independent learning replicate or new teacher population. Cross-run compositions are not equal-total-training-budget methods; nominal old-history KL matching is not equality of trajectory distributions. The residual includes policy-conditioned training/data effects; this experiment does not isolate moving upper as its cause. Source-minus-zero remains inconclusive. Stage87/88 positives, Stage89 lower negatives, Stage67 critic-credit HOLD and the closed frequency-superiority claim remain unchanged.

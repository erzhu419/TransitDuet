# Stage100 Fresh Staged Upper: Full Result

All nine tasks t128955-t128963 completed with exit0 on node004/005/006. Native qualification and the frozen 28-contrast aggregate matched the official summary. All 32 registered final upper checkpoint files exist.

## Primary Result
All four registered primary contrasts have positive corrected CI lower bounds and are positive on every one of eight new teacher roots. staged_confirmation = supported.

| Registered endpoint | Mean reward difference | Corrected CI |
| --- | ---: | --- |
| 50/staged_common_minus_source_common | 0.373130 | [0.252867, 0.471483] |
| 50/staged_common_minus_staged_independent | 0.418844 | [0.105664, 0.790777] |
| 100/staged_common_minus_source_common | 0.582006 | [0.415282, 0.762972] |
| 100/staged_common_minus_staged_independent | 2.161257 | [1.059829, 3.884574] |

The unchanged 65,536-draw equal-root bootstrap / Bonferroni28 family contains 14 positive, 7 negative and 7 inconclusive endpoints. No old-cohort results were pooled.

## Claim Boundary
Both newly learned uppers improve over their respective U0/fixed-lower compositions. Most common-route superiority is inherited lower contribution: source-common minus source-independent is +0.430134 / +2.224272 at periods50/100, compared with total staged-route gains +0.418844 / +2.161257.
With the same common lower, common-trained upper is worse than the crossed independent-trained upper: -0.003811 CI [-0.006823,-0.000637] at50 and -0.016892 CI [-0.048526,-0.000489] at100. Common-trained upper is also worse under the independent lower at both periods. Thus matched-upper specialization is not supported.
At100 both staged uppers are worse than fixed Stage98 UJ with the same lower: common -0.095812 CI [-0.155695,-0.026478], independent -0.076605 CI [-0.134953,-0.021158]. Corresponding comparisons at50 and staged-versus-original-joint comparisons at both periods are inconclusive. Retain all seven negative endpoints.

Exact incremental cost: 22,016 native episodes / 26,419,200 steps, 256 upper-mean updates, 48 donor loads and 32 server-only final checkpoints. Source/std/value/Adam/decoder and learned-lower freezes passed; replay/critic/forecaster fits and native traces remain zero. Native runtime: 633.95-656.93 s/root. Saved compact JSON: 64747 bytes before pretty-printing. Preparation costs remain separate.

## Next Step
Stop repeating this U0-start test. Freeze a UJ-initialized upper-refinement test with both registered U0-trained lowers unchanged. Make improvement over fixed UJ with the same lower the four primary endpoints; keep crossed-actor specialization diagnostics and separate the extra refinement compute. No donor, root, period or radius selection.

Scope: new-teacher staged MC mean learning on the same native PointMaze task, not full actor-critic, unseen-task generalization, matched-upper coordination or frequency-superiority proof.

# Stage89: learned upper contribution is supported

t121890-t121898 all done/exit0; eight roots, 3,584 episodes /4,300,800 steps; mechanical gate passed. Wall98-104s/root, peak4171-4225MiB. Only47,440-byte compact JSON pulled.
Read32 registered Stage88 final checkpoints server-side; 112 exact actor compositions. Zero policy updates, critic/forecaster fits, checkpoint writes or traces. Both periods pass checkpoint/std/value/source-Adam/decoder/forecaster freeze and paired-noise checks.
U0/L0 are source actors, UJ/LJ joint-call actors, LL lower-only learned lower. All22 endpoints share the frozen equal-root bootstrap65536 / Bonferroni22 family on fresh Stage89 evaluation seeds.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| Direct upper: UJ/LJ minus U0/LJ | +0.2428 [0.1663, 0.2997] | +0.4683 [0.3472, 0.6215] |
| Transfer: UJ/LL minus U0/LL | +0.2422 [0.1660, 0.2990] | +0.4657 [0.3455, 0.6192] |
| Upper on source lower: UJ/L0 minus U0/L0 | +0.2943 [0.1982, 0.3659] | +0.6085 [0.4648, 0.7516] |
| Lower difference, fixed U0: U0/LJ minus U0/LL | -0.02735 [-0.03525, -0.01839] | -0.02841 [-0.04374, -0.01440] |
| Lower difference, fixed UJ: UJ/LJ minus UJ/LL | -0.02673 [-0.03431, -0.01803] | -0.02578 [-0.04236, -0.01194] |
| Joint minus lower-only: UJ/LJ minus U0/LL | +0.2155 [0.1464, 0.2703] | +0.4399 [0.3224, 0.5849] |
| Upper-by-lower interaction | +0.000621 [0.000246, 0.001065] | +0.002630 [0.000805, 0.004840] |

The frozen direct-upper gate passes in both periods; direct and transfer gains are positive at all8 roots. The joint lower deficit is negative at all8 roots with either fixed upper. The supported interaction is small: 0.26%/0.56% of the corresponding direct-upper mean gain.
Stage87/88 joint-versus-lower wins are retained, and Stage89 attributes the net gain to upper improvement overcoming a small lower deficit. This is not evidence that jointly learned lower is better than separately learned lower.
Next: preregister a frozen-upper lower baseline with exactly joint-call's lower budget (.00099/.000995), keeping Stage88 training rosters and evaluating on fresh seeds, to distinguish budget reservation from joint-training effects; no budget search or checkpoint adoption.

## Limitations
Same eight teachers and fixed Stage88 checkpoints, not new independent teacher populations, retraining counterfactuals, full actor-critic or frequency superiority. Cross-run actor compositions are not equal-training-budget methods. Lower-only retains an active frozen upper; source-minus-zero CIs remain inconclusive. The lower deficit's cause is not yet identified. Stage67 critic-credit HOLD and the closed frequency-superiority claim remain unchanged.

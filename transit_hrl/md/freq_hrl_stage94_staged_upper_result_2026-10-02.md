# Stage94: staged-learning gate supported

t126885-t126893 all done/exit0; mechanical gate passed. Eight roots, 22,016 additional episodes / 26,419,200 steps; wall 611-696 s/root, peak 4525-4664 MiB. Pulled 142,002-byte compact JSON; checkpoints remain server-side.
Both Stage93 U0-trained lowers stayed frozen; each new upper started from U0, used independent upper/lower noise credit, and was evaluated only after update 8 on 32 fresh paired episodes/root/period.

| Primary | period 50: mean [CI]; positive roots | period 100: mean [CI]; positive roots |
| --- | --- | --- |
| New upper benefit: staged-common minus source-common | +0.21766 [+0.15420,+0.26980]; 8/8 | +0.41254 [+0.28195,+0.55147]; 8/8 |
| Matched staged route: common minus independent | +0.36793 [+0.00959,+0.68177]; 7/8 | +1.05964 [+0.24876,+1.91987]; 7/8 |

All four preregistered primary lower CI bounds are positive: staged_confirmation=supported. All 28 equal-root bootstrap 65,536 / Bonferroni28 CIs were independently recomputed; 15 positive, 1 negative, 12 inconclusive. The 256 upper updates, 32 final checkpoints, 48 donor loads, 32 initialization checks and 176 compositions match preregistration; zero replay, critic fits or native traces, and every upper gradient used 64 episodes.
Independent-lower upper learning also improves reward at both periods: +0.22527 and +0.45402, both positive supported.
Actor swaps attribute most common-route superiority to the inherited lower: common-upper minus independent-upper with the same common lower is -0.00451 at 50 and -0.01389 at 100, both inconclusive. Common-stage upper transferred onto the independent lower at 100 is negative supported (-0.02161).
Staged-common exceeds old Stage88 joint at100 (+1.22331 [+0.19588,+2.55893]; 7/8), but not conclusively at 50. Neither staged upper is shown to outperform the fixed old UJ with the same lower. Retain all secondary results.
Next: independently confirm the frozen staged protocol with new upper-training/evaluation samples, then test unseen teachers; keep Stage94 separate from confirmation and do not retune radius, roots, periods or endpoints.

## Limitations
Same eight teachers and inherited Stage93 lowers, not teacher-population replication; small dense-return gains. Route superiority is not uniform: root 310101 is negative at50 and root 310011 at100; a crossing-zero CI is not equivalence. Mean-only, fixed-std/decoder learning is not full actor-critic, joint end-to-end HRL or frequency superiority. Inherited lower compute is separate. Stage93's global failure, Stage67 HOLD and earlier negatives remain unchanged.

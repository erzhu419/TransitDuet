# Stage95: independent upper-sample confirmation supported

t126913-t126921 all done/exit0; mechanical gate passed. Eight roots, 22,016 additional episodes / 26,419,200 steps; wall 624-649 s/root, peak 4550-4711 MiB. Pulled 142,789-byte compact JSON; checkpoints remain server-side.
Both uppers were retrained from U0 on new Stage95 training/evaluation samples; no Stage94 upper reuse. The same Stage93 lowers stayed frozen, with independent upper/lower credit noise and final-update-only evaluation.

| Primary | period 50: mean [CI]; positive roots | period 100: mean [CI]; positive roots |
| --- | --- | --- |
| New upper benefit: staged-common minus source-common | +0.22880 [+0.17288,+0.31061]; 8/8 | +0.43080 [+0.29035,+0.55426]; 8/8 |
| Matched staged route: common minus independent | +0.37161 [+0.04809,+0.60262]; 7/8 | +1.03171 [+0.43282,+1.65026]; 8/8 |

All four primary lower CI bounds are positive: staged_confirmation=supported. All 28 equal-root bootstrap 65,536 / Bonferroni28 CIs were independently reproduced, with 15 positive, 3 negative and 10 inconclusive. The 256 upper updates, 32 checkpoints, 48 donor loads, 32 initialization checks and 176 compositions match preregistration; every upper gradient used 64 episodes, with zero replay, critic fits and native traces.
Stage94 and Stage95 independently pass the same four-primary gate on separate upper-training/evaluation samples; their effects and CIs are reported separately, not pooled. Stage94 means were +0.21766/+0.41254 for upper benefit and +0.36793/+1.05964 for route benefit at 50/100.
Independent-lower upper learning also remains positive supported at both periods (+0.23783/+0.47004). With the same common lower, common-stage upper minus independent-stage upper is -0.00351/-0.00587, both inconclusive: most route superiority is inherited lower contribution.
Common-stage upper transferred to the independent lower at 100 is negative supported (-0.01211 [-0.01742,-0.00660]; 0/8). Original U0 plan versus zero is negative supported at both periods (-0.31796 [-0.50432,-0.10124]; 1/8; -0.89457 [-1.37364,-0.28429]; 1/8); retain these three negative endpoints.
Staged-common exceeds old Stage88 joint at 100 (+1.48599 [+0.09124,+3.44355]; 7/8), but not conclusively at 50. Neither staged upper is shown to outperform fixed UJ with the same lower.
Next: stop same-teacher seed repetition; preregister unseen-teacher validation with unchanged learning rules and explicit zero/no-plan, U0, fixed-UJ and matched-independent controls. Report upstream teacher/lower training compute separately.

## Limitations
Confirmation is conditional on the same eight teachers and fixed Stage93 lowers, not fresh-lower or teacher-population replication. Gains are small in dense-return units and route superiority is not uniform (root 310101 remains negative at 50). Crossing-zero CI is not equivalence or noninferiority. Mean-only fixed-std/decoder learning is not full actor-critic, joint end-to-end HRL or frequency superiority; Stage93's global failure, Stage67 HOLD and earlier negatives remain unchanged.

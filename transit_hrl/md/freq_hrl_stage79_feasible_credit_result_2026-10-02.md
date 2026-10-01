# Stage79 Result and Next Step

Preflight t120340/341 and full t120368-376 all completed with exit0. Code revision
bc4bf17c0d; five focused tests passed. All objective, common-noise, fixed-KL and
budget checks passed; source actors/Adam stayed unchanged. Full budget:512 fresh
credit episodes plus3072 independent evaluation episodes,4,300,800 native steps.
All32 actor-direction geometries passed, exact raw Gaussian KL0.00099849-0.00100152.
Root compute time102.1-104.9s. No optimizer/critic/forecaster fits, traces or checkpoints.

All18 Bonferroni-corrected root-bootstrap reward CIs cross zero:

| Period / Actor | MC plus minus source, CI | MC plus minus MC minus, CI |
| --- | --- | --- |
| 50 / upper | -.00262 [-.02985,+.02254] | -.00468 [-.05920,+.04596] |
| 50 / lower | -.05123 [-.37322,+.28264] | -.06866 [-.69486,+.59127] |
| 100 / upper | -.03223 [-.09671,+.02824] | -.06199 [-.18772,+.05699] |
| 100 / lower | -.01510 [-.50517,+.56775] | +.02596 [-.94828,+1.20364] |

Descriptive A/B gradient cosine means: upper50 +.2002 (6/8 positive roots), lower50
-.0605 (2/8), upper100 -.0741 (3/8), lower100 +.0399 (5/8). Fresh native MC directions
are not consistently stable, and no tested positive direction has supported reward gain.
The source-minus-zero contrasts are also inconclusive; this does not erase Stage78's
period100 harm result or establish noninferiority. Stage67 HOLD remains.

Next isolate scenario variation using independent action-noise rollouts of the same
native scenario and a leave-other-rollout-out time-aligned MC baseline. Treat scenario
groups, not dependent trajectories, as gradient-noise units. Keep alpha/radius fixed;
no policy adoption or longer joint training yet. [Compact evidence](../results/pointmaze_feasible_credit_stage79_full_20261002_r1/qualification_compact.json).

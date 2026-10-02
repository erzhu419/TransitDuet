# Stage90: fixed lower-budget attribution

Stage89 supported direct learned-upper gain, but LJ lost to LL at both fixed uppers. Add LM, trained with source upper frozen and exactly joint-call's lower KL: .00099 at period50, .000995 at period100. No allocation search or checkpoint adoption.
Only LM is newly trained. Reuse Stage88 update8 UJ/LJ and U0/LL checkpoints server-side, with frozen Stage78 teachers/decoder/std/values/Adam/forecaster. Eight rounds,64 episodes per round; exact Stage88 full training scenario/noise rosters, not a new independent training replication.
Evaluate eight variants on32 fresh Stage90 paired seeds per root/period, horizon1200: U0/L0, zero, U0/LM, U0/LL, U0/LJ, UJ/LJ, UJ/LM, UJ/LL. Preflight: one root, horizon300, two rounds using the first two Stage88 full scenarios per batch, four new evaluation seeds; no reward gate or adopted preflight weights.

## Frozen Analysis
All28 endpoints use one equal-root bootstrap65536 / Bonferroni28 family, seed(90,90090), same eight teachers. No cross-stage population pooling or CI-driven retuning.
At fixed U0: budget component = U0/LM minus U0/LL; matched-budget training component = U0/LJ minus U0/LM. Their sum is the original lower gap U0/LJ minus U0/LL. Repeat this decomposition at fixed UJ using the same lower actors.
Report each component's signed CI at both periods. A CI crossing zero is inconclusive, not absence, equivalence or no-harm. Do not declare the whole deficit explained by budget merely because the matched-budget training CI crosses zero.
Also report upper gains with each lower, joint-minus-lower-only, and zero controls. Cross-run compositions are diagnostic interventions, not new equal-training-budget methods.
Full additional cost: 12,288 native episodes /14,745,600 steps; 128 lower-mean updates,16 new final lower checkpoints,32 Stage88 checkpoint loads,128 exact compositions. No upper training, critic/forecaster fits, Adam steps, intermediate selection or raw traces.
Scheduler dynamically uses node001-006, no pin;9CPU/8192MiB with8 workers per full root, preflight3CPU/3072MiB with2 workers. Final checkpoints remain server-only; pull only completion markers and compact JSON.

## Limitations
Conditional diagnosis on the same teachers and intentionally paired Stage88 training rosters, not an independent learning replicate, full actor-critic or frequency-superiority result. Nominal old-history lower KL matching is not equality of final trajectory distributions or total joint budget. Stage87/88 positives, Stage89 lower negatives, Stage67 critic-credit HOLD and the closed frequency-superiority claim remain unchanged.

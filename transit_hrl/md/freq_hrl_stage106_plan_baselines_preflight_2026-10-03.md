# Stage106 Plan-Baseline Native Preflight

t131256/t131257 finished with exit0. Native qualification and read-only reaggregation exactly match the official summary. All312 native episodes/93,600 steps,24 actor-mean updates,72 extra upper replay forwards and variant-specific planning counters match the frozen budget. Both baselines use the full union of the joint learners' credit rosters. Forecast-only has no upper inference; flat has no upper, plan or forecaster calls. Native wall79.590s; no checkpoints or raw traces were written/pulled.

Single-root, short-horizon primary reward differences are50:conditioned-minus-forecast +0.028741, conditioned-minus-flat -40.047177;100:-0.102354 and -69.394012. These descriptive results have no CI and do not decide admission. All26 contrasts and own-initialization controls are retained in the16,396-byte compact pull.

## Scope
The failed r1 native cleanup call is archived, its queued analyzer cancelled and full r1 never dispatched. The repaired r2 uses the same seeds, task, sample budget and endpoints; only cleanup now follows the existing task.environment.close() API. All7 focused tests passed after that repair. This remains teacher-assisted, fixed-std MC learning, not full actor-critic or from-scratch flat PPO.

Full r2 dispatched unchanged:t131845-t131852 ran two roots each on node001/004/005/006, with dynamic node001-006 eligibility and no hard pin; t131853 performed all-root qualification. All nine tasks subsequently finished with exit0. The [full result](freq_hrl_stage106_plan_baselines_result_2026-10-03.md) retains the not-supported learned-plan increment claim:joint-versus-forecast is inconclusive and joint-versus-flat is supported harm at both periods. Preflight remains mechanical evidence, not performance admission.

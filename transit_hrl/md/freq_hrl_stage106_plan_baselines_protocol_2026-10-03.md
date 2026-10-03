# Stage106 Learned-Plan Value Against Trained Baselines

Stage105 supported lower common-upper-noise conditioning under0.4-0.8s regime dwell, with small reward gains. Its `zero` control retained the ridge forecast:it did not establish learned-upper or hierarchical value against a trained flat controller.

Stage106 keeps the shifted task, eight original Stage96 initializers, Stage97 decoder/forecaster, periods50/100 and8 mean updates. It trains four methods:joint independent, joint conditioned, forecast-only lower and genuine flat lower. Joint methods use64 upper plus64 lower credit paths/update; both lower-only methods use the union of those rosters,128 lower paths/update. Every method receives nominal primitive-call-weighted KL0.001/update. Extra replay and differing gradient/Fisher work are counted separately.

Forecast-only executes the unchanged causal ridge reference and planned velocity, with no upper inference or learned residual. Flat uses physical state, current target error, full64-step measurement history and latest causal target-velocity difference; it has no upper inference, forecaster, reference renewal or plan input. Both retain the original392-dimensional lower architecture and weights. Period is only the initializer index for flat. Frozen critic-only clocks do not enter the flat actor or MC credit.

Seven evaluation policies include the three distinct untrained initializers and four final learners. These initial controls expose the flat input-semantics change without refitting or selecting a new teacher. Each policy gets32 fresh independent paired evaluation paths/period, after exactly the final update.

All26 final/initial reward contrasts form one equal-root bootstrap family:65,536 draws, Bonferroni26, seed(106,106106). Four primaries are conditioned joint minus trained forecast and minus trained flat, at both periods. Only four positive corrected lower CI bounds support the registered plan-increment claim. Preflight is mechanical only, not a reward admission test. No cohort pooling, intermediate evaluation, selection or seed extension.

Full cost:69,120 native episodes/82,944,000 steps,768 active actor-mean updates,512 update operations,73,728 extra upper replay forwards and64 final server-only checkpoints. Preflight:312 episodes/93,600 steps,24 mean updates and no checkpoints. Scheduler placement is dynamic node001-006,3CPU/3GiB for preflight and9CPU/8GiB per full root. Only completion markers and compact JSON are pulled.

## Scope
This compares teacher-assisted fixed-std MC actor-mean learners, not from-scratch flat PPO/SAC, matched parameter counts, full actor-critic or frequency superiority. All source models, values, Adam, std and forecast/decoder fits stay frozen. Stage67 HOLD and Stage104/105 results remain separate.

Completed:the [full eight-root result](freq_hrl_stage106_plan_baselines_result_2026-10-03.md) does not support the learned-plan increment claim. Conditioning still helps relative to independent joint learning, but neither period establishes conditioned joint superiority over trained forecast; trained flat is significantly better at both periods. The mandatory forecast-reference route is not a positive hierarchy claim; next diagnose optional planning with unchanged full lower feedback and retain the strong baselines.

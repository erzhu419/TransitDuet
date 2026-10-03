# Stage107 Optional Planning Above Strong Flat Feedback

Stage106's mandatory forecast-reference route is not supported against trained forecast and is worse than trained flat. Keep its negative result and the unchanged0.4-0.8s regime task. This stage tests optional advice, not a new stress setting or a weaker lower controller.

All arms start from every Stage106 final flat lower, with the corresponding frozen conditioned-joint upper. Keep the first392 flat features intact:physical state,current target error,64-step history and causal instantaneous target velocity. Append4 advice features:reference minus current target and planned minus causal target velocity. Zero-pad the actor and critic input weights so the initial lower function ignores arbitrary advice; no teacher/forecaster/decoder refit.

Train three lower-only methods:blind,forecast hint and frozen learned-plan hint. The blind arm uses4 zeros with no planner calls. Every arm receives128 native paths/update,8 updates and nominal lower KL0.001/update; both periods and all8 roots are retained. Upper,std,values and Adam stay frozen. Actual forecast/upper computation is extra and reported.

Evaluate6 fixed policies on32 fresh paired paths/period:initial blind,three trained methods,and the forecast/learned lower with advice zeroed and all planning disabled. All20 contrasts form one equal-root bootstrap family,65,536 draws,Bonferroni20,seed(107,107107). Require6 positive primary corrected lower CI bounds:learned hint minus blind,forecast hint,and its own blinded execution at both periods. Preflight is mechanical only; no reward admission,selection or seed extension.

Full cost:52,224 new native episodes/62,668,800 steps,384 lower-mean updates and48 final server-only checkpoints. Preflight:240 episodes/72,000 steps,12 updates,no checkpoints. Scheduler placement is dynamic node001-006;3CPU/3GiB preflight,9CPU/8GiB per full root. Pull only completion markers and compact JSON. Stage96 source preparation and the complete Stage106 campaign remain separate inherited costs.

## Scope
This is conditional lower adaptation to fixed upper advice above a reused strong flat policy, not joint-HRL training,unseen-task confirmation or frequency superiority. The20-contrast family evaluates this changed interface; it does not pool with or erase Stage106. Stage67 critic HOLD remains unchanged.

Native preflight passed: [result](freq_hrl_stage107_optional_plan_preflight_2026-10-04.md). The frozen [full result](freq_hrl_stage107_optional_plan_result_2026-10-04.md) is not_supported:tiny advice gains,but no confirmed learned increment over forecast. Next isolate fixed-lower learned residual content without more training or task changes.

# Stage76: historical velocity support and frozen actor response

Stage75 attributes most sampled-residual execution loss to planned velocity. Stage55 BC labels use deterministic zero-mean upper actions, while later native probes sample residuals. Stage76 tests input coverage and conditional actor response before changing the planner.

- Calibration uses all original Stage55 teacher-label episodes: 2 per period in preflight, 8 per period/root in full. Reconstruct the actual392-dimensional BC state, compare recorded reference/velocity with the base curve, and reproduce the saved clone's final tanh-command MSE. No warmup, training, evaluation reward or reward-selected scale enters calibration.
- Output physical calibrations: label velocity coordinate ranges, speed q99 and position-target error q99. These describe historical input envelopes; they are not yet applied as action constraints.
- Replay original Stage75 native plan inputs using its seeds, proposed upper actions, target-driver prefixes and world bounds. Match recorded reference-target, reference-residual and velocity-residual energy before interpreting input coverage. Load layout once without simulator reset/step. No new native rollout, trace or checkpoint.
- Report native base/residual input fractions outside label coordinate ranges and above label speed q99. Four residual-minus-base endpoints (two metrics x periods50/100), all reported with65,536 equal-root bootstrap draws and two-sided Bonferroni4. Preflight has no CI claim.
- Actor response uses historical label states with the first390 columns fixed. Compare recorded velocity, zero velocity and sampled-residual velocity under the frozen clone; report conditional Gaussian KL, tanh-command RMS change and BC MSE. The sampled proposals use a new fixed Stage76 RNG and the source upper Gaussian marginals, not a reconstructed native RNG stream. These response summaries are descriptive.
- Full audit: 8 frozen roots,128 label archives /153,600 label states,512 existing native plan frames,1,536,000 replayed velocity rows,460,800 offline actor rows and768 actor forward batches. Optimization, forecaster fitting and native steps are zero.
- Scheduler: dynamic node001..006, no pin; each root and qualifier1CPU/2048MiB. Preflight root310001 runs before the full eight-root matrix. Pull only completion markers and compact JSON.

Preflight r1: t118825 failed before label audit because the layout probe closed the task instead of its owned environment; t118826 cancelled. Fixed in b32c0b9822 with an ownership/API test (7 tests passed). Retry r2 retains the same calibration inputs, four endpoints and statistical protocol; r1 provides no scientific result.

Preflight r2: t118827/t118828 both done0 on node006;1,200 exact label states,2 saved BC MSE reproductions,8 Stage75 plan-energy frame matches,7,200 replayed velocity rows and3,600 conditional actor rows. Native steps, fitting and optimizer steps are zero. Mechanical gate passed; full eight-root audit released without performance selection.

Full r1: t118830..t118837 dispatched dynamically on node001/004/005/006; all eight roots done0. t118838 qualifies the four frozen coverage endpoints after completion markers are synchronized.

## Limits

Coordinate and speed envelopes are not joint-state support. Historical-state response is not an on-policy reward counterfactual. This audit calibrates inputs and tests a mechanism hypothesis; it does not repair the learned upper, adopt a budget, or establish frequency superiority. Stage67 HOLD remains unchanged.

# V72: full-day action-increment scorer, whole-day-excluded development

Freeze actor, upper/B200 deployment, dual, production values and main. Reuse
only V70's balanced 12 consumed demand days: 144 prefixes, 32 futures each,
seven actions. No new simulation or V71-only context selection. Keep V67/V69
out of this first balanced fit rather than mixing different sampling designs.

Inputs: complete original encoded state, completed-prefix CV-squared and
current actor probabilities. No day/seed/context identifiers, RNG, future
return/launch times or labels enter features. Pool the four fixed actor contexts.
Fit five outputs: passenger person-seconds, terminal CV-squared, mean readiness
delay, unlaunched and incomplete trips, all relative to the native current action
within each paired future, then averaged over all 32 futures.

Primary model: linear state interactions with action and squared action,
`[a-a0, a^2-a0^2] x [1, standardized_state]`, actions normalized by 45 seconds.
This preserves exact native zero and contrast antisymmetry/transitivity while
allowing an interior optimum. Fixed ridge 1 on the day-balanced mean loss;
closed-form float64 solve, no hyperparameter or stopping selection. Controls:
zero differences and the same two-term action-only model. Report effective
parameters; full encoded state is retained, including pending commands.

Twelve leave-one-demand-day-out fits: exclude all actors/windows/futures of
that day from fitting and standardization. Score all six nonnative actions;
report mean32-label MAE/MSE/RMSE, sign/order diagnostics and paired day CIs.
Select only from predictions: minimize passenger difference with all four
predicted constraint differences <=0; ties keep native. Evaluate fixed choices
on every excluded day's 32 futures. Report day700006 concentration and its
removal without refitting or retuning. Never choose a model from these results.

Development prediction gate: primary MAE and MSE contrast CI uppers <0 versus
both controls for all three continuous outputs. Joint gate: passenger CI upper
<0, all four constraint uppers <=0. Both are required before considering fresh
day/prefix confirmation, not deployment. V70 gates remain unchanged.

Server smoke: focused tests, full cache preparation and first held-day fit
(reused formally), then 12-worker folds and aggregation through scheduler on
node001-node006. Save all fit/cache arrays only there; pull small JSON/markers.
Save the all-day development fit separately, never in the production trainer.

## Outcome / Next Step

Source `f3eb5126b6`; scheduler `t165745` smoke (11 tests and actual first fold),
`t165746` 12-worker folds, `t165747` aggregate on node004. All computations
completed; 12 whole-day-excluded folds, no new rollout or controller update.
Only 25,180 bytes of markers/summary JSON were pulled. Primary has 2,420
coefficients, effective degrees of freedom about 11.7-11.9 per output.
Passenger MAE: zero 1796.49, action-only 1740.49, state 1997.77 person-seconds.
State-minus-zero MAE +201.28 [90.32,320.85]; CV-squared MAE also worsens,
+5.77e-6 [1.80e-6,1.10e-5]. Readiness MSE improves versus zero by 0.00794 s^2,
but its MAE CI spans zero. Both development gates fail. State changes 23/144
actions: six improve passenger exposure, 16 worsen, one ties; two have jointly
nonworse point estimates. Mean exposure delta +4.85 [-94.62,97.16]
person-seconds; omitting evaluation day700006 gives +39.95 [-33.45,120.92].
Do not start fresh confirmation or promote action-only as a posthoc winner.

Next test causal state sufficiency: recapture current full-fleet AVL and
published dispatch commitments on the original frozen native references,
requiring prefix/day identity before reusing their existing action labels.
Compare with this fixed scorer on whole-day-excluded data, not by tuning ridge
or gates. Current encoding includes ready counts and pending lower commands,
not full fleet positions/operating states or complete dispatch commitments;
whether that omission explains the failure remains a hypothesis.

Limitations: consumed development labels, overlapping cross-fit training sets,
32-mean label noise and four actors from two original training seeds. This is
not new-day confirmation or proof of fixed-headway superiority. Current actor
probabilities only partially describe differing continuation policies. Labels
evaluate single interventions, not jointly executing choices across windows.
Removing day700006 from reported effects does not remove it from other folds'
training sets.

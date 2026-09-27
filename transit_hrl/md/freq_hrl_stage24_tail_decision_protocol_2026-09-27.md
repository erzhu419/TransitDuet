# Stage-24 Tail-Credit Decision Value

Freeze roots 209011/209061, their 16 selected states, Stage-16 short-window
contrasts, Stage-21 ordered 64-future caches and Stage-23 predictions. No fits,
new samples, root expansion, model selection or deployment. Preflight 208001
uses its existing two states/four futures.

Primary split: replicas 0-31 select the oracle-assisted action; replicas 32-63
score it. Reverse these halves for sensitivity only. Preflight uses 0-1/2-3.
Never choose a split or direction from the results, pool overlapping directions
as independent observations, or use scoring labels to select the oracle action.

Let w be observed 50-step ISE(wait)-ISE(now), and A the cached continuation
contrast. Short-window reference chooses now iff w>0. Oracle-tail chooses now
iff w+mean(A_selection)>0; frozen linear/contextual/random-context diagnostics
choose now iff w+predicted_tail>0. Zero ties choose wait-one-check.
For action indicator d and reference d0, score (d-d0)*(w+A_scoring).
Positive is local ISE reduction versus the short-window reference.

Report all-state and path-wise benefits, action/switch counts and oracle action
agreement between directions. Conditional Monte Carlo SE is
sqrt(sum_i (d_i-d0_i)^2 * sample_var(A_score_i)/K_score)/N. Report directions
separately, without a new qualification gate, confidence claim or p-value.
Stage-23's failed MSE gate remains unchanged regardless of these diagnostics.

Use scheduleurm on dynamic node001-node006, one CPU/1.5 GB per task. Only
three cached result inputs and compact JSON outputs; zero new environment
steps, controller training, critic fits or policy updates.

Nineteen focused tests passed, including disjoint action selection/scoring,
negative oracle benefit, cost-sign algebra, Monte Carlo SE, tie behavior,
credit reconstruction, frozen inputs and scheduler resources.

Execution at `b67004479e`: preflight `t101520` on node006, full tasks
`t101523/101524` on node006/node001, all completed. The
[result](freq_hrl_stage24_tail_decision_result_2026-09-27.md) records the
unchanged split and all three frozen critic comparisons.

## Limitations

Both references use the counterfactual short-window outcome, not an online
observation. The finite-sample oracle is not a certified upper bound. These
are reused-development, fixed-state contrasts under the frozen continuation,
not additive episode improvements or independent policy validation.

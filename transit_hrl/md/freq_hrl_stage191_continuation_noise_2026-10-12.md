# Stage191: Same-State Continuation Noise

Stage190 tasks t144436-t144650 are all done/exit0. Its calibrated selectors
fall back everywhere. This does not distinguish representation failure from
single-continuation label noise, and does not establish that all upper learning
is ineffective. Do not weaken its quantile after inspecting the test outcomes.

Reuse Stage187 replica0 in both roots and five regimes, at probes8/22/36:
30 existing states in ten scene groups. For each state, eight future seeds
replace ScenarioTape only after the exact preaction state/RNG check. Keep the
episode demand profile, peak phase, OD multipliers, duration prior and frozen
lower. Each draw executes an untouched original replay, a resampled greedy
observer and six matched candidates, including exact identity. All suffixes
resume the original greedy selector. Read frozen Stage188/190 actions before
rollouts; do not select them using the new outcomes.

Analyze pairwise effect power with finite-draw variance removed, odd/even-half
rank stability and cross-half choice outcomes, existing-label oracle transfer,
and frozen-policy outcomes under the eight new innovations. Oracles remain
diagnostics. If ordering is stable but critic selection is wrong, prioritize
joint bus-plan representation. If sampling changes ordering and estimated
benefits, prioritize expected-continuation targets before another value fit.

Budget:240 cells/1,920 full episodes, zero training updates or new prefix scenes.
First qualify ten probe22/draw0 cells across both roots/all regimes, then230
remaining cells. Use scheduler nodes001-006, unbound, one CPU/3GB per cell.
Full rollouts, draw labels and the original bank stay server-only; sync compact
receipts and the final report. This is conditional Poisson/speed-innovation
diagnosis with fixed episode-level latent profiles, not full regime uncertainty,
new performance confirmation or end-to-end HRL.

Run `native_transit_continuation_noise_stage191_development_20261012_r1`,
source `b5bd1f4228`; preflight tasks `t144777-t144786` started on nodes001/004/005/006.
All ten preflight tasks finished exit0 with full qualification. Eight targeted
synthetic/mock tests pass, including advancing without downloading receipts.
Remaining230 are now eligible. No manual result pull was needed for preflight.
They are registered as `t144798-t145027`. The submitter now uses the scheduler
results API's terminal status; that API does not expose an exit-code field.

## Result and Next Step
All240 native tasks completed exit0. Report task `t145090` qualified1,920
episodes,240 exact source replays,240 resampled identities and1,680 matched
preaction state/RNG comparisons. Only one171,110-byte report was pulled.

Half-choice agreement is10/30; cost rank concordance is254/449 (56.6%).
Within-half oracle cost improvement -0.006576 becomes cross-half +0.004637.
The original single-label oracle changes from -0.004463 to +0.009249 under
the independent future draws. Early probe8 is particularly unstable: its
oracle transfer is +0.028100. Even the eight-draw mean's estimated sampling
variance exceeds debiased effect power by1.72x for cost and2.01x for wait.

Frozen Stage190 sequence choices have negative new mean cost/wait in both
roots on scene/lower-root splits, but regime/root401 wait remains +0.001683.
These are30 development states in ten prefix groups, not the600-state gate
or independent performance confirmation. Prioritize expected-continuation
targets before adding encoder capacity or changing the gate.

# V100: native learning-contract audit

V99 is complete: all 40 passive replays exactly reproduce cached endpoints and
action counts. Retrieved only the 170,248-byte summary. The upper runs (262
callbacks, 88 plans/day), but mean path RMS target adjustment is only 1.146 s
with preserve and 1.056 s with zero holding; maximum preserve adjustment is
4.528 s. Thus V98 concerns weak deployed upper authority, not disabled planning.

Preserve's two-term shadow regularity-minus-observed-load reward favors holding
by +0.007579 reward units/decision; its expected holding is 14.230 s. About
35.76% of chosen actions have negative immediate two-term advantage over zero.
This is neither the full reward nor evidence that the critic is wrong. The
full shaped reward improves much more than these two terms explain, while
V98 shows worse passenger time. Audit the base headway reward before tuning
the load coefficient.

V100 prepare joins the 40 server-side V99 cases, verifies the additive reward
identity and reports physical/reward contrasts over 20 actor/day cells. Native
B200 is then replayed on the same ten actors and two consumed days, r0: 20
paths, smoke reused. No preserve-mixture continuation is used to judge the
native critic. The original local replay states, terminal flags, gamma, alpha,
dual, target aggregation and clipping are retained. Actual runner reward scale
is 1, despite the trainer function's default 10.

Measure actor KL/TV to the entropy-aware cost/pessimism Boltzmann law; critic
residuals to its exact clipped training backup; and reward/cost calibration
to native per-trip Monte Carlo returns. Soft returns include future entropy,
not current-action entropy. Verify actual replay next-state chains, native
probabilities, cached endpoints/action counts, and smoke's full timing path.
Also record raw reward to identify any additional shaping term explicitly.

Decision: objective mismatch warrants system-credit redesign; substantial actor
or critic errors warrant a separate learning fix. Do not change all three at
once. This diagnostic selects neither an action nor a coefficient. Actors,
critics, dual and main remain frozen. Tests, join, replays and aggregation run
only through scheduler on node001-node006; retrieve small summaries only.

Limitations: two already-consumed demand days, not a fresh efficacy test.
Bellman consistency is not action-value truth. Finite native soft-return
errors include continuation noise and state aliasing; they do not establish
counterfactual action rankings or system passenger-cost calibration.

Source `fbfb9f9d9d`. Prepare `t175920` completed on node004: all four focused
tests pass and the V99 reward identity holds. Retrieved only its 19,465-byte
summary. Native smoke `t175921` gates five CPU-batch shards `t175922-t175926`
covering [0,2), [2,7), [7,12), [12,16), [16,20), using 2/5/5/4/4 workers.
Aggregate `t175927` waits for all five shard markers. No V100 calibration
result is claimed before those paths and native-contract checks complete.

Native smoke passed on node004: exact cached endpoints/action counts and full
timing path, one-to-one decision/transition identity, native probabilities and
all local-trip next-state chains. The raw headway reward plus regularity minus
load exactly reconstructs shaped reward, with no unexplained penalty. This is
instrumentation admission, not evidence of actor/critic correctness. The smoke
row is reused in the formal 20-path budget.

# V110: causal observed nonholding-disturbance compensation

One new structural candidate, not a gain search. Keep the V106/V107 bilateral
feedback gain g=.25 and seven feasible commands. At a valid current ready
decision, use the matched predecessor's already-completed departure, actual
launch times and holding executed since each launch:

    R = forward_gap - release_gap - executed_holding_gap
    D = (follower_gap - forward_gap)/2
    desired_holding = g*D - (1-g)*R

With missing follower evidence, D=target-forward as in the unchanged baseline;
missing forward evidence keeps zero. Quantize after combining the unrounded
terms, using the original 0--45 second cap, feasible mask and shorter-bin ties.
The fixed .75 feedforward complements the .25 feedback under a local additive
nonholding-disturbance model. It is a hypothesis, not an established optimal
coefficient. No gain/window/threshold fitting. R includes observed travel and
stop service; no future arrival, current trip's later stop, return time or label
is consulted. Reuse the existing completed-prefix coordinate implementation.

Use V106 development actors 12037/933083 and four new demand days
2101001--2101004. Controls: native B200, preserve, zero, bilateral025. Native
and preserve have three paired samplers; the other three have one shared
deterministic path per actor/day: 72 full-day rollouts. Upper/deployment state,
fleet, clocks, scenarios, actors, critics and duals stay fixed.

Report all paired service effects, crossed actor/day intervals after sampler
averaging, completion, and feedforward sign/magnitude/cap/actual command dose.
Command deltas compare the shadow unmodified rule on the candidate's own
trajectory; they are not matched-dose causal mediation.
Diagnostic success requires passenger upper <0 and actual CV2/readiness/
unlaunched/incomplete upper <=0 against native, preserve AND bilateral025.
Zero only describes the service/regularity tradeoff. The two-actor pilot cannot
admit a controller or establish frequency separation/fixed-baseline superiority.
A successful candidate needs separate independent confirmation; failure ends
this fixed candidate, without coefficient rescue or adaptive day extension.

Smoke uses unit tests, four exact cached V106 control paths and two sampler
replays of the new candidate on the same consumed development context. Fresh
day shards wait for the real smoke marker; aggregation waits for actual shards.
All computation goes through scheduler on node001--node006 using the isolated
CPU environment. Timing primitives and checkpoints stay server-side; only
markers and the final small JSON synchronize. Main policy remains frozen.

## Run Record

Frozen source `de2f0f81ae`; the output-counter serialization fix does not change
the control equation. Smoke `t183624` passed on node006: eight tests, four exact
cached V106 controls and two sampler-invariant candidate paths with exercised
command changes. No new-day labels were used. Formal `t183625--t183629` covers
0--5/5--21/21--38/38--52/52--72, with 5/16/17/14/20 workers from live CPU
capacity (72 parallel workers, one batch). Aggregate `t183631` waits for those
five actual shard markers. Only the 200-byte smoke marker returned before
formal dispatch; all tasks are restricted to node001--node006.

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

## Outcome

All five shards and aggregate completed with exit 0. Retrieved only the
43,943-byte final JSON. The candidate fails all three registered joint gates:

| FF minus control | Journey, min (95% CI) | Actual CV2 (95% CI) | Readiness, s (95% CI) | Holding, s/trip |
| --- | --- | --- | --- | --- |
| B200 | +4.550 [3.218, 6.100] | +.3482 [.2619, .5086] | +587 [426, 769] | +47.36 |
| preserve | +4.933 [3.549, 6.638] | +.3520 [.2667, .5054] | +702 [559, 862] | +62.55 |
| bilateral025 | +6.169 [4.421, 8.274] | +.3503 [.2678, .5069] | +1004 [855, 1148] | +185.37 |

Both actor means and all four demand-day means worsen for all these endpoints.
Even zero holding has better actual CV2 (.1789 versus .4044). All arms finish
all trips: this is a control failure, not a failed rollout or uncleared service.
Relative to bilateral025, waiting exposure increases 6.253 million person-s and
onboard exposure 1.313 million person-s. Holding rises from 154.21 to 339.58
s/trip; the extra holding does not buy regularity or readiness.

Among 38,056 valid-forward decisions, commands differ from the shadow original
rule on the candidate's own states in 58.62%. Unrounded requests are <=0 in
44.42% and >=45 in 28.15% (72.57% combined). These are request-boundary rates,
not executed-bin fractions. Mean signed feedforward is only +1.29 s, but mean
absolute feedforward is 60.20 s and mean quantized command change is +6.21 s.
Missing forward evidence is 3,544/41,600 decisions. The shadow comparison is
not a matched-dose intervention or a causal decomposition of the service loss.

Close this fixed candidate without gain rescue or further confirmation. V111
will audit existing action/timing records and the persistent-disturbance
response before proposing a different controller. Main stays unchanged.

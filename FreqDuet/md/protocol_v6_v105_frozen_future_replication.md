# V105: independent futures, frozen V104 action choices

Answer one question before more training: do fixed system-action gains repeat?
Keep V104's 16 prefixes (two archived actors, four consumed demand days, two
decision times). Add 32 independent futures per prefix, two predetermined
16-future cohorts. Seven executable holding bins: 3584 full-day continuations.
Namespace: `205000000 + context * 100 + future`; no reuse of V104's innovations.
This is new-future replication, not new-prefix/day/actor generalization.

Before any new labels, snapshot the V104 model/normalizers, references, model
forecasts and registered A/direct-Q/native laws on the server. Reconstruct and
freeze the two original teacher proposals from old futures 0--1 and 2--3. Check
that their old crossed evaluation reproduces the completed V104 review. The
primary teacher summary is their fixed equal mixture; each original component
is diagnostic only. No selecting the better component, refitting, reweighting,
new margin, new gate, or scalar goal tuning on V105 labels. Keep native actors,
critics, duals, upper plan and main unchanged. Only the anchor action is forced;
subsequent upper/lower feedback remains native and fully closed loop.

Report all 21 signed action-pair differences, old/new contrast agreement,
independent-cohort cross-products, MC noise and frozen-model difference error
against direct Q and zero A. Evaluate every frozen law against paired seven-bin
full-system labels. Primary laws: separated A and equal-mixture teacher, not a
post-hoc winner. Costs: once-only full-day passenger time, actual arrival CV
squared, actual readiness delay; also journey, holding and unfinished service.

For each cohort and pooled 32 report crossed actor/day 95% bootstrap intervals
after averaging futures/time, and separate fixed-prefix MC bootstrap intervals
(5000 draws, paired futures resampled within prefix, all actions/laws together).
Future count never becomes a count of independent demand days. Joint benefit
requires passenger upper < 0 and CV/readiness/unlaunched/incomplete upper <= 0
under both pooled intervals and nonharmful point estimates in both 16-future
cohorts; report reversals rather than hide them. Independent-cohort contrast
cross-product with CI lower > 0 identifies repeatable action signal; noise is
reported separately, not clipped or used to rescue a fit.

Decision after completion only: stable contrasts + teacher joint benefit but
model failure -> investigate supervised state mapping; unstable contrasts ->
fix label estimation before actor training; teacher still lacks joint benefit
under more precise independent evaluation -> pause this single-action route
for these states/bins and reassess intervention/service tradeoffs. A failed
teacher is not proof that the whole control problem is insoluble. Even a pass
does not authorize main promotion or imply additive full-day policy gains.

Scheduler only, node001--node006, isolated worktree and dedicated CPU environment.
Smoke: unit tests, freeze, then reproduce V104 context-0/future-0 extreme actions
and event timings exactly; no new future labels in smoke. Formal shards wait for
smoke, aggregate waits for all actual shard markers. Models and raw records stay
on server; synchronize only small markers and final JSON. No training expansion.

## Run record

Frozen source `77b3274a09`. Smoke `t183471` passed on node004: five tests,
V104 model/law snapshot, reconstructed old teacher evaluation, and both cached
extreme-action paths exactly reproduced. No new future labels used in smoke.
Formal shards `t183472--t183475`: ranges 0--881, 881--1880, 1880--2468,
2468--3584; 15/17/10/19 workers, respectively. Soft-preferred nodes 002/004/005/006,
eligible on 001--006 only. Aggregate `t183476` waits for all four small markers.
No additional training or main-policy change.

The last shard started later and remained active at the next review. Preserve
2969 finished arms, stop its old process group, and reshard only missing IDs
2969--3584 via the live CPU planner; the arm grid, source and RNG seeds remain
unchanged. Aggregate waits for actual replacement markers and reads all arms.
Before seeing effects, correct the diagnostic noise of the half-mean difference
to Var(first)/16 + Var(second)/16 (the initial field divided their sum by 32).
Rename it `split_difference_noise`, with a regression test. This reporting fix
does not affect any model, proposal, rollout, cross-product or benefit gate.

Replacement tasks `t183492--t183495` cover 2969--3200, 3200--3328,
3328--3413 and 3413--3584, with 26/15/10/19 workers. The cancelled
`t183475` process group was confirmed empty before submission. Aggregate
`t183476` now waits for the three original completed markers plus these four
replacement markers, not the cancelled shard. Its reporting source is
`e932f6e3ab`; all rollouts retain `77b3274a09`. Six unit tests, including the
noise regression, must succeed on the compute node before aggregation.

## Outcome

All 3584 arms and six tests completed; aggregate `t183476` on node004.
Only the 430434-byte final JSON and small markers returned locally.
A and teacher mixture fail crossed, conditional-MC and two-cohort joint gates.
A passenger change = -1073 [-4776,3972] person-s, CV2 = +0.00000896
[-0.000252,0.000192], readiness = +0.597 [-0.448,1.681] s (crossed intervals).
Conditional readiness harm is positive: [0.115,1.101] s. Teacher passenger =
-185 [-3114,3747], CV2 = +0.0000249 [-0.0000463,0.0000826], readiness =
+0.239 [-0.570,1.285]; passenger changes sign across the two cohorts.
A contrast MSE is significantly worse than zero for all three goals, though
better than direct Q. Independent-half cross-products confirm readiness signal
(1.365 [0.064,3.172]), not passenger (57.7m [-4.6m,140.4m]) or CV2
(-4.61e-10 [-4.46e-8,5.46e-8]). All three teacher optimism intervals are positive.
Do not refit/select on V105 or expand training. Next: V106 full-day service
tradeoffs with frozen controls and simple feedback; main remains unchanged.

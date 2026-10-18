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
No V105 effect result yet. No additional training or main-policy change.

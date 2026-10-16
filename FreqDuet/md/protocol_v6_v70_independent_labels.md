# V70: independent balanced-prefix label confirmation

Freeze main, the four V59 updated actors, upper/B200 deployment, dual and all
value models. No fitting or controller promotion. Use the unchanged V69 event
coupling, seven holding actions and exact completed-prefix contract.

Predeclare demand seeds 700001-700012, disjoint from V64-V69 development and
validation days. Four actors x 12 days x all three targets (7200/21600/36000s):
48 full-day references, 144 label-independent first-eligible prefixes.
Private pre-prefix sampler seed: `7000000 + case*1000 + day_index`.
Each prefix uses 32 new futures, seed `70000000 + context*100 + future`;
seven arms share that future and event uniforms. Total: 4,608 grids / 32,256
action replays, plus 48 references. Every day contains all three windows.

Select on futures 0-15 and evaluate on 16-31, then reverse and average.
Selection minimizes passenger exposure subject to mean CV-squared, readiness,
unlaunched and incomplete trips no worse than native; ties preserve native.
Only held-out halves enter the primary selection effect. No threshold tuning
or action reselection using evaluation outcomes.

Report signed noise-corrected signal squared, single-future and 32-mean SNR,
16/16 sign agreement, held-future passenger/constraint deltas, and descriptive
window/training-seed/day breakdowns. Bootstrap 12 demand-day clusters after
averaging actors and windows within day. Passenger unit: full-day person-seconds.
Independent-label gate: passenger and CV-squared corrected-signal CI lower >0
and 32-mean SNR CI lower >1. Joint-opportunity gate: passenger delta CI upper <0
and all four constraint delta CI uppers <=0. Report both gates separately;
neither changes deployment automatically.

Server smoke: affected V67-V69 tests, V70 tests, one cached V69 native endpoint
identity, two new references, three complete grids spanning all windows/two
original training seeds, and three forced-native full-path identities.
27 smoke replays; 23 reused formally. References/matrix/aggregation are gated
through scheduler on node001-node006 only. Pull small JSON/markers/logs only.

Limitations: fresh prefixes and futures, but the same four actors from two
original training seeds; not new-training-seed or deployable-controller
confirmation. Pending arrivals and realized day profiles stay conditional.
Per-prefix selection uses counterfactual labels, not an online observable scorer.
Focused context selection below is posthoc development diagnosis, not confirmation.

## Execution

Source `b3259d1997`, immutable snapshot `source/v70_independent_20261016`.
Smoke `t165690` passed on node004: 31 tests and all 27 replay checks.
References `t165693` completed with rc=0/retry=0: 48 trajectories, 144 prefixes;
two smoke references reused. All four matrix shards `t165694-t165697` and
aggregate `t165698` completed with rc=0/retry=0: 32,256 action replays.
Focused audit `t165705` (source `34a083af33`) passed two tests and completed
without added replay or fitting. Only 23,876 bytes of result JSON were pulled.

## Outcome And Decision

Action signal repeats: passenger/CV-squared corrected signal is positive on
10/12 and 9/12 days; half-sign agreement is 70.1%/66.6%. Mean32 SNR is 3.791
[0.685,7.157] / 0.896 [0.122,2.516]. The independent-label gate did not pass.

Held-future passenger delta: -821.61 person-seconds [-2441.58,39.43];
CV-squared: -6.22e-6 [-1.59e-5,1.37e-6]; readiness delay: +0.00331s
[-0.00202,0.00958]. Unlaunched/incomplete deltas are zero. Of 82 changed
half-decisions, 44 improve exposure, 37 worsen it, one ties; 19 have jointly
nonworse held-half point estimates. The joint-opportunity gate did not pass.

Keeping choices fixed, secondary journey delta is -0.000650 minutes
[-0.001934,0.000035], about -0.039s per passenger, not a confirmed benefit.
Day 700006 supplies 92.3% of gross and 96.7% of net exposure improvement.
Without it, exposure delta is -29.81 [-145.11,60.99] over 11 days; corrected
action signal remains positive. Nonzero action effects and useful joint
improvement are different findings.

Retain event-coupled label tools and keep main/actor/dual/value frozen.
Next diagnose fixed within-prefix interventions at c15 (5->45s), c123
(0->20/45s), contrasting c51/c87 constraint failures and c125 (30->0s).
Trace downstream holding, waiting and vehicle returns before fitting a global
scorer or increasing another full future matrix. Do not alter either V70 gate.

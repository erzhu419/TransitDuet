# Stage154: Learned Adaptation Versus Constant Dispatch Phase

Freeze Stage153 physical-L1 checkpoints299, roots313/331. Reproduce all twenty
source baseline episodes per root before intervening. Compare learned upper
against a source-wide actor-call-weighted mean command and fixed 5/7/9 s delays.
Reuse the source zero-upper results. Keep lower policy and execution unchanged.

Two scheduler jobs, 100 episodes each, 12,276,000 native ticks, no training.
All node001-006 eligible, one CPU/3 GB per job, no checkpoint download.
The 45 focused tests pass; live source reproduction remains a worker prerequisite.
Run `native_transit_phase_adaptation_stage154_frozen_20261010_r1`: t141001/root313
and t141002/root331 completed on node006/node005. All forty source baselines
reproduced exactly; 200 frozen episodes completed with zero training updates.
Report each learned-minus-constant and constant-minus-zero contrast, with wait,
reward and fleet-cost components. Do not select a retrospective winning constant.
If phase explains the gain, the next implementation targets temporal credit/plans;
if adaptation adds value, validate it on independent scenes before expanding seeds.

## Result

Learned-minus-constant-mean cost: `+0.020831/+0.000211` for roots313/331;
reward: `-0.10235/-0.29710`; restricted wait: `-0.00020/+0.00160` min.
Learned-minus-fixed7 cost: `+0.020831/+0.000682`, reward `-0.10235/-0.67695`.
Learned-minus-fixed9 cost/reward are also worse for both roots. These results
do not support adaptation beyond a constant departure phase.

Stop L1 tuning/seed expansion. Next isolate delayed upper credit: current service
intervals last about 180 s, but dispatch effects extend over a route journey;
one-step RE-SAC must propagate this through bootstrapping with only ten upper
updates per episode. Test an off-policy-corrected multi-step backup against the
same one-step learner, retaining physical execution, reward and network geometry.

## Limitations

These constants were chosen after Stage153; this is mechanism diagnosis, not an
independent performance comparison or evidence of domain-general learned HRL.

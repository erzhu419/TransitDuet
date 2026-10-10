# Stage154: Learned Adaptation Versus Constant Dispatch Phase

Freeze Stage153 physical-L1 checkpoints299, roots313/331. Reproduce all twenty
source baseline episodes per root before intervening. Compare learned upper
against a source-wide actor-call-weighted mean command and fixed 5/7/9 s delays.
Reuse the source zero-upper results. Keep lower policy and execution unchanged.

Two scheduler jobs, 100 episodes each, 12,276,000 native ticks, no training.
All node001-006 eligible, one CPU/3 GB per job, no checkpoint download.
The 45 focused tests pass; live source reproduction remains a worker prerequisite.
Report each learned-minus-constant and constant-minus-zero contrast, with wait,
reward and fleet-cost components. Do not select a retrospective winning constant.
If phase explains the gain, the next implementation targets temporal credit/plans;
if adaptation adds value, validate it on independent scenes before expanding seeds.

## Limitations

These constants were chosen after Stage153; this is mechanism diagnosis, not an
independent performance comparison or evidence of domain-general learned HRL.

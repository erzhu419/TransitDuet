# V160: metadata and independent-day reservation

Completed server task `t204621`, source `14d930ee25`, node006/eight workers.
Eight tests and all 384 actor/day pairs / 768 paths passed.

Sidecars cover 3,993,600 queries, completing 35,462 conditional post-ready
supports and shadow service support for 3,343,763 structured queries; all 1,695
changed service means reconcile. They preserve V159 residual files and add
separate shadow service support, ready-cached follower-validity and distinct
query/ready/policy/label clocks. Conditional means and shadow service differences
must reproduce the recorded values. Late policy load remains unavailable at
query time. The opaque full-residual cohort remains unallocated.

Reserved calibration days: `2851001-2851032`; confirmation: `2852001-2852032`.
Each group has 32 whole demand days, all 12 archived actors and both old/handoff
modes: 768 paths per group. Samplers are `1600000000+i` in calibration-then-
confirmation day order. Checks of 10,921 metadata files and 327 generator
scripts found no day/sampler conflicts. Registered days remain unconsumed.

This preregistration reserves the split and frozen collection contract only.
Next: seal the interval estimator and final inference gates before fresh
collection; confirmation remains separate from calibration. No collector is
submitted here. V151/main, V146(3s/0s), 3600s window and .1/.1 tails stay fixed;
no replay, training or interval fit. Raw sidecars stay server-side.

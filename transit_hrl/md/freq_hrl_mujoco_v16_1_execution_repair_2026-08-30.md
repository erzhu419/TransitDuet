# MuJoCo v16.1 paired audit-gauge execution repair

## Invalidated v1 execution

The first `mujoco_v16_1_audit_gauge_paired_preflight_v1` dispatch is not
scientific evidence. Its nine reward-anchor tasks completed or were already in
flight, but every continuation task released by a completed anchor stopped at
configuration validation before its first training update. The core correctly
rejected a nonzero paired reference-reduction target without an active
deployment-frequency level constraint. The remaining queued continuations were
cancelled. No reward or frequency outcome from a continuation was inspected.

The v1 launcher also exposed a second execution-contract limitation during a
local continuation smoke: the paired checkpoint loader assumed every anchor had
router strength zero. That assumption was valid for the earlier homotopy
experiments but invalid for a reward-compatible anchor trained entirely in the
full audit-gauge coordinates.

## v2 repair

The repaired protocol makes two source-bound changes:

1. The candidate uses active upper and lower deployment-frequency duals for its
   five-percent paired targets. The compute-matched reward continuation keeps
   both targets and deployment dual rates at zero.
2. Paired checkpoints now serialize, load, and validate the anchor router
   strength. The v2 anchor and both continuations use
   `causal_audit_aligned_gauge` at strength one.

A one-iteration local anchor-to-candidate continuation completed with
`status=valid` after both repairs. The v2 scheduler signature and run namespace
are distinct from v1. The frozen seed roots may be reused because the v1 defect
prevented every continuation training update and no v1 performance outcome was
used to choose the repair. This remains development evidence, not confirmation.

## Claim boundary

The invalidated v1 run supports no algorithm, reward, leakage, or generalization
claim. The v2 protocol can support only its pre-registered paired mechanism gate
after all cells complete and the independent analyzer accepts them.

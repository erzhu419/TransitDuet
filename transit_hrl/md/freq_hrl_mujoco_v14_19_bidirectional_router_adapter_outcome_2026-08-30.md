# Freq-HRL MuJoCo v14.19 Bidirectional Router Adapter Outcome

## Execution

All nine frozen cells in
`mujoco_v14_19_bidirectional_router_adapter_screen_20260830_r1` completed.
Scheduler tasks `t84681` through `t84689` ran as independent one-core jobs on
`node005`. The preregistration records source revision
`f284551783778e9124021eb52e4c7dd1757d9fbe`.

The first sync message mislabeled the run as "v14.18 cells" because the shared
launcher contained a hard-coded display string. The task signatures,
preregistration, grid, analyzer, and result paths were all v14.19. The display
string was corrected after outcome access.

## Frozen decision

The bidirectional router adapter mechanism was not supported. The identical
selector found an eligible candidate in six of nine cells:

- HalfCheetah: 3/3 supported.
- Hopper: 3/3 supported.
- Walker2d: 0/3 supported.

All 99 grid evaluations had zero reward violations. Among the six selected
candidates, four used strength `0.6` and two used `0.7`. Their relative merit
reductions ranged from 39.60% to 60.00%, with median 57.54% and mean 54.80%.

## Walker2d boundary

Router strength `0.5` minimized frequency-violation merit in every Walker2d
cell. Both directions were worse. Strength `0.6` changed relative merit by
`-76.26%`, `-136.12%`, and `-2.95%` across the three seeds. Strengths below
`0.5` were substantially worse; at `0.4`, the relative changes were
`-2938.38%`, `-1675.63%`, and `-2594.11%`.

This result rejects the hypothesis that a function-preserving routing
coefficient alone can break the v14.17 restoration deadlock across the three
MuJoCo environments. The negative result is consistent across all Walker2d
replicates and is not repaired by searching the full `[0, 1]` interval at 0.1
resolution.

## Next mechanism

Further router-grid tuning is not justified. The remaining failure requires an
actor-level change that alters latent upper/lower policies while controlling
executed behavior. The next design must first audit the earlier joint learned
projection, anchor replay, trust-region, and restoration experiments, then
introduce a coupled deployment-aligned actor intervention only if it addresses
their observed failure modes.

## Claim boundary

This adaptive development result establishes a router-family boundary, not a
paper performance claim. Reward preservation held for the tested routing grid,
but router-only frequency restoration did not generalize to Walker2d.

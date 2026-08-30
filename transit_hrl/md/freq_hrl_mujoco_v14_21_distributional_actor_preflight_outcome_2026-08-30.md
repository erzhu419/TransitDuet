# MuJoCo v14.21 distributional actor preflight outcome

## Execution

Run `mujoco_v14_21_distributional_actor_preflight_20260830_r1` completed as
scheduler tasks `t84726` through `t84728` on `node003`. Each task requested 16
CPU cores and 8 GB RAM with `require_node=null`; no Slurm path was used. The
preregistration records source revision
`e83a55fdd640513b9b4a744193ce0d12b58f8f30`. HalfCheetah was the longest cell
at about 342 seconds and used 6.835 GB peak RAM.

## Frozen decision

The preflight is **not supported**. Two of three independent validation cells
passed, below the frozen 3/3 gate.

| Environment | Eligible design candidates | Design merit | Validation merit | Validation reward violations | Validation result |
|---|---:|---:|---:|---:|---|
| HalfCheetah-v5 | 3 | 0.007035 | 0.087560 | 3 | not supported |
| Hopper-v5 | 9 | 0.055321 | 0.055344 | 0 | supported |
| Walker2d-v5 | 9 | 0.055112 | 0.054929 | 0 | supported |

The common baseline merit was about `0.055402`. Hopper and Walker2d validation
reductions were `0.105%` and `0.854%`, respectively, but all 20 registered
frequency constraints remained violated in each environment. These are weak
merit improvements, not restored frequency feasibility.

HalfCheetah selected the negative ranked-antithetic gradient at output-head RMS
`1e-7`. Its 64-path design merit fell by 87.3%, but on 64 independent paths the
merit rose by 58.0%, the frequency violation count was 15, and three reward
floors failed. The larger root ensemble therefore reduced estimator noise
without making the local actor-head direction transferable across occupancy
realizations.

## Mechanism boundary

The distributional frozen-output-head restoration mechanism is rejected. The
result does not authorize new roots, alternate mode aggregation, more random
directions, or another step grid on the same local-search family.

The next admissible algorithmic change must alter the training objective. In
particular, deployment-frequency control needs an action-conditioned cost
critic trained on current closed-loop occupancy, so deterministic actor updates
can account for how actions change future state visitation. Existing native
cost value critics estimate state cost under sampled rollouts, while the
deployment projection differentiates deterministic actions on frozen states;
neither supplies this action-to-future-cost gradient. Any such replacement must
pass a new fresh-root preflight before entering multiseed development.

This is adaptive development evidence only and supports no manuscript efficacy
claim.

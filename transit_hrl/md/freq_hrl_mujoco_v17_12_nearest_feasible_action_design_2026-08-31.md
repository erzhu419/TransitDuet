# MuJoCo v17.12 Nearest Feasible Action Design

## Purpose

V17.11 closed router-only development on the frozen 120-path panel. No causal
fixed-total-action candidate passed the recovery gate, and seven Hopper paths
were already proven infeasible for any bounded full-horizon responsibility
split. The remaining question is how much the learned total action must change
before actor-level development is justified.

## Convex Target

Starting from each v17.6 oracle component pair, v17.12 computes its Euclidean
projection onto the intersection of:

- the exact causal HPF8 upper-energy ball;
- the exact causal LPF32 lower-energy ball;
- the registered upper and lower component boxes.

Dykstra's algorithm is used because every set has an exact projection. The
quadratic frequency-ball projection is solved by eigendecomposition and scalar
multiplier bisection. Already feasible paths are exact zero-change fixed points.
For the seven actor-floor paths, a second diagnostic adds the nominal
environment total-action box; it is not used to select the training target.

## Frozen Gate

The reused panel must reproduce 113 feasible paths and seven actor-floor paths.
All 120 frequency-only targets must be feasible, the 113 feasible references
must remain unchanged, and all seven actor-floor targets must change total
action. The maximum per-path total-action correction is limited to RMS 0.05 and
absolute component 0.25. Passing authorizes grouped causal actor-target
distillation on reused paths. Failure rejects this local-correction route.

The target arrays remain on node003. Only JSON summaries and a server-location
marker are synchronized. This acausal reused-path oracle is development-only;
it cannot support online-policy, reward, fresh-seed, or manuscript performance
claims.

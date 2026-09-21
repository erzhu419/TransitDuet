# PointMaze Exogenous-Control Stage-5 V1 Result

Date: 2026-09-21

## Outcome

The frozen development run
`pointmaze_exogenous_stage5_v1_development_20260921_r1` completed all 16 cells:
two methods and eight independent optimizer roots. It contains 256 final
held-out episodes and 256 root- and seed-paired untrained episodes. The
registered ordinary-HRL substrate gate is **not supported**, so frequency
routing remains blocked.

All tasks `t99962` through `t99977` completed without failed or cancelled
cells. Compact `result.json` files were synchronized from node003 through
node006, with four cells from each node.

The execution audit passed:

- all 16 method/root signatures occurred exactly once;
- all train, selection, and evaluation roles matched the frozen fresh seeds;
- every episode had 300 transitions and finite outputs;
- every HRL episode had 12 upper decisions and 12 lower option boundaries;
- flat, upper, and lower states were all 134-dimensional;
- flat PPO used 67,973 trainable parameters and HRL used 68,424;
- each flat cell performed 12,288 gradient updates and each HRL cell 18,432;
- current physical feedback, causal external observability, 0.12 force RMS,
  and 0.04-second force periods matched the protocol;
- external target vertices and directions were exactly paired across methods;
- runtime versions were identical across all cells.

## Registered Results

Root-level means and 95% t intervals were:

| Method | Tracking success | Episode return | Tracking RMSE |
|---|---:|---:|---:|
| flat external history | 0.548 [0.300, 0.796] | 186.960 | 0.577 |
| HRL external history | 0.616 [0.424, 0.808] | 191.612 | 0.562 |

The hierarchical final-minus-untrained contrasts were positive:

| Endpoint | Mean improvement [95% CI] | Status |
|---|---:|---|
| tracking success | +0.443 [0.246, 0.640] | supported |
| episode return | +78.754 [48.385, 109.123] | supported |
| tracking RMSE reduction | +0.747 [0.488, 1.006] | supported |
| final-distance reduction | +1.247 [0.888, 1.605] | supported |

The absolute HRL success interval lower endpoint was 0.424, below the frozen
0.50 threshold. The conjunctive gate therefore failed even though both paired
learning-gain conditions passed.

HRL-versus-flat effects were inconclusive: success improvement was +0.068 with
95% CI [-0.313, 0.449], and return improvement was +4.652 with 95% CI
[-47.136, 56.440]. These comparisons were registered as descriptive rather
than gating.

## Failure Diagnosis

Seven HRL roots reached held-out success between 0.499 and 0.879, while root
134127 reached only 0.151. Its validation success stayed near 0.10 throughout
training; the frozen success-first selector chose iteration 191 even though
dense validation return continued improving through iteration 767. Root 134113
ended at the boundary with success 0.499. This is optimizer/training stability,
not a malformed external path: route identities were method-paired and all
causal and physical contracts passed.

The result does not authorize adding roots to rescue the interval or proceeding
to frequency routing. A fresh development repair should target optimization
stability and checkpoint alignment, then use new optimizer and role seeds.

## Claim Boundary

Allowed: on the frozen separate-exogenous PointMaze task, ordinary HRL learned
substantially relative to its own untrained policy in tracking success, return,
RMSE, and final distance.

Forbidden: Stage 5 V1 passes the ordinary-HRL substrate gate, proves a hierarchy
advantage over flat PPO, supports frequency routing, or validates a general
Freq-HRL claim.

Machine-readable analysis is under
`results/pointmaze_exogenous_stage5_v1_development_20260921_r1/analysis/`.


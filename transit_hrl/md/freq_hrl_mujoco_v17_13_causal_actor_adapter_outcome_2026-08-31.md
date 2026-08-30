# MuJoCo v17.13 Causal Actor Adapter Outcome

## Decision

Status: `causal_actor_adapter_stops_before_fresh_path_access`.

The 900-member frozen screen used eight leave-one-seed-out folds. Its
three-ranking prefilter sent 48 candidates to exact full-horizon responsibility
oracles. The selected W8, ridge `1e-4`, actor-floor weight 256, gain 1.0, cap
0.01 adapter kept all 120 paths valid and preserved all 113 paths already
feasible. It changed the post-clipping action on all seven actor-floor paths and
met the target-fidelity and trust-region gates, but recovered only 3/7 paths:
1/2 for seed 2802248628 and 2/5 for seed 294864529. No fresh path was accessed.

## Frontier

All 48 exact-oracle candidates preserved 113/113 reference-feasible paths.
Thirty-two recovered two actor-floor paths and sixteen recovered three; none
recovered more. Thus the evaluated mild-adapter frontier does not satisfy the
actor-floor gate.

The boundary is narrower than the full 900-grid. Because the frozen prefilter
ranked globally by target fidelity and preservation, every exact-oracle
candidate had gain 0.5 or 1.0. Gain 1.5 and 2.0 candidates remained unexamined
by the expensive feasibility oracle even though their OOF target errors were
finite and some met the target-fidelity threshold. V17.13 therefore stops fresh
validation but authorizes a separately frozen gain-stratified reused-path audit.

## Efficiency

Path-level sufficient statistics and 32 bounded oracle workers reduced worker
runtime to 30.8 seconds and scheduler wall time to 49.1 seconds. Targets stayed
server-only on node003; only summary and model JSON files were synchronized.

## Claim Boundary

This is grouped reused-path causal-adapter development. It does not establish
reward improvement, online policy learning, fresh-seed generalization, leakage
no-tradeoff, or manuscript support.

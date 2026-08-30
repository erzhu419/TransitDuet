# MuJoCo v17.6 Full-Horizon Responsibility Oracle

## Purpose

V17.5 showed that minimizing the current lower-budget violation can increase
future FIR floors and worsen full-trajectory leakage. The next decision must be
based on the globally best responsibility split for a fixed learned total-action
trace, not another local proxy.

## Convex Program

For a frozen total action trace `a[1:T]`, choose upper responsibility `u[1:T]`
and define lower responsibility `l = a - u`. The component box is exact:

```text
max(-U, a_t - L) <= u_t <= min(U, a_t + L).
```

Let `H8 = I - R8` and `L32 = R32`, where `Rw` is the exact causal truncated
rolling-mean matrix used by the registered diagnostics. The oracle solves:

```text
minimize    ||L32 (a - u)||^2
subject to  ||H8 u||^2 / (T d) <= upper_budget^2
            component box constraints.
```

This is a convex quadratically constrained box problem. Its Lagrangian
subproblems are bounded linear least squares and are solved with SciPy BVLS. A
shared nonnegative multiplier is bracketed and bisected until the upper budget
is met. The output reports the upper-constrained lower floor, the independently
minimum upper power, BVLS optimality, box KKT residual, component-bound
violation, and reconstruction error.

## Interpretation

For each rejected v17.4 path:

- if the oracle meets both budgets while v17.4 does not, the online router is
  the limiting mechanism;
- if the oracle cannot meet both budgets, the learned total action is physically
  incompatible with the fixed component budgets on that path;
- mixed path outcomes require both an online-router change and an actor-level
  feasibility constraint.

The oracle is acausal and is never deployed as the algorithm. It is a mechanism
diagnostic and an upper bound on what any function-preserving responsibility
router can achieve on the frozen trace.

## Evidence Boundary

The first oracle run reuses rejected v17.4 paths and is development-only. It may
select the next mechanism but cannot support a performance or generalization
claim. Any selected online method must be frozen before fresh optimizer and
evaluation seeds are generated.

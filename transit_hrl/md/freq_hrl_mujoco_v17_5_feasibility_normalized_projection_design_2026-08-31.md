# MuJoCo v17.5: feasibility-normalized audit projection

## Status

This document defines the v17.5 mechanism before checkpoint replay or fresh
held-out execution. V17.5 is a feasibility and attribution repair. It is not a
reward-improvement claim and must not be described as a new behavioral result
unless a later learned-policy experiment changes the executed total action.

## Motivation

V17.4 exactly preserved 120 paired control paths and reduced lower-LPF32 and
normalized joint merit in all three MuJoCo environments. It failed the frozen
absolute contract because HalfCheetah and Hopper remained above the fixed lower
budget and HalfCheetah sometimes had no physically feasible upper component
that met the current upper budget. A fixed pair of component budgets is not a
meaningful algorithmic target unless their joint physical feasibility is known.

## Causal feasible envelope

For one action coordinate at primitive step `t`, v17.5 constructs three closed
intervals for canonical upper action `u_t`:

1. `P_t`: the component interval implied by upper/lower action bounds and the
   exact realized total `a_t = u_t + l_t`.
2. `U_t`: values whose current causal HPF8 upper residual is within `0.075`.
3. `L_t`: values whose current causal LPF32 lower residual is within `0.0475`.

The intervals use the exact last seven canonical upper actions and last 31
canonical lower actions. They therefore require no future action and match the
registered online audits.

The projection is lexicographic:

1. If `P_t` intersects `U_t`, the upper budget is hard. Otherwise choose the
   point in `P_t` with minimum unavoidable upper-budget violation.
2. Within that upper-feasible domain, if `L_t` intersects, enforce both budgets
   and minimize the same 16-step receding audit objective as v17.4.
3. If the lower intersection is empty, choose the point with minimum current
   lower-budget violation conditional on the upper contract.

This produces an independently computable causal floor. The router reports the
joint-feasible rate, unavoidable upper and conditional-lower violation RMS, and
normalized budget-excess regret. Full-strength regret must be zero within
numerical tolerance.

## Invariants

The projection retains the v17.4 invariants:

- exact total-action reconstruction;
- physical component bounds;
- strength-independent canonical histories and complete FIR policy state;
- paired reward, action, and latent-policy trace identity;
- online HPF8/LPF32 residuals equal batch diagnostics.

The new floor is conditional on the realized causal history and the
upper-first responsibility contract. It is not a noncausal global trajectory
oracle and does not prove that a learned policy could not change its total
action while preserving return.

## Development sequence

1. Unit and random-sequence tests must reproduce the interval construction
   independently of the implementation.
2. A diagnostic replay may reuse the rejected v17.4 checkpoints on their
   server, but it is development-only and cannot support a fresh-seed claim.
3. If replay shows avoidable excess, revise the projection before fresh roots.
4. If replay shows near-zero excess, the remaining problem is policy-level
   feasibility: train the actor under a reward floor to reduce the unavoidable
   physical floor, then evaluate on fresh optimizer and held-out roots.

No multiseed expansion is permitted solely because the feasibility diagnostics
are implemented.

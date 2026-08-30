# MuJoCo v17.4: causal streaming audit projection

## Status

This document freezes the v17.4 mechanism and its preflight decision rule. It
is a development protocol, not evidence that the manuscript claim is
supported. Expansion beyond the registered preflight is allowed only if the
preflight passes every gate below.

## Why v17.3 was insufficient

The registered v17.3 audit-optimal macro gauge preserved reward, action,
latent, and transition traces exactly, but it failed the frequency gate in all
three MuJoCo environments. Its optimizer compared the candidate with an
internal held-canonical plan rather than the actually executed raw
factorization. It also froze a finite plan at macro boundaries, while the
registered HPF8 and LPF32 diagnostics are streaming filters whose memory spans
those boundaries. The resulting projection was locally optimal for a surrogate
that did not match the measured intervention.

## Frozen mechanism

At primitive step `t`, let the realized total normalized action be

`a_t = u_t + l_t`.

The router preserves `a_t` exactly and chooses only its responsibility split.
It stores the last seven canonical upper actions and the last 31 canonical
lower actions, which are the complete finite-memory state for the registered
causal HPF8 and LPF32 diagnostics. No future action or environment state is
used.

For a 16-step receding horizon, v17.4 holds the newly observed total action
constant as a forecast and solves the one-coordinate quadratic induced by

`sum_tau [(HPF8(u)_tau / 0.075)^2 + (LPF32(l)_tau / 0.0475)^2]`,

subject to the physical upper/lower component bounds. The current-step upper
HPF budget is imposed as a hard constraint whenever its intersection with the
physical interval is nonempty. If it is physically infeasible, the projection
chooses the physically feasible upper component with minimum unavoidable
current upper residual and reports the violation. Only the first coordinate is
executed; the horizon is rebuilt after the next realized total action.

The canonical histories are updated from total action alone and do not depend
on intervention strength or the supplied raw factorization. Therefore the
strength-zero and strength-one paired rollouts receive identical policy state.
The policy observes all 38 right-aligned FIR history vectors plus two normalized
valid-history counts.

## Mechanical invariants

For paired strength-zero and strength-one evaluation from one frozen
checkpoint, every path must satisfy:

1. Identical rewards, observations, actions, latent actions, and transition
   traces within numerical tolerance.
2. Exact pre-split total-action reconstruction.
3. Identical complete filter-state inputs across intervention strengths.
4. Upper and lower component bounds at every executed step.
5. Batch HPF8/LPF32 residuals equal the online residuals emitted by the router.

Any failure is an implementation failure and blocks interpretation of frequency
metrics.

## Registered preflight

The preflight uses one optimizer cell for each of `HalfCheetah-v5`,
`Hopper-v5`, and `Walker2d-v5`, with four training seeds, four checkpoint
selection seeds, eight fresh paired evaluation seeds, and five registered
disturbance modes. Training occurs at strength zero with the complete v17.4
state; evaluation replays the same frozen checkpoint at strengths zero and one.
No v17.4 hyperparameter arm is selected from these evaluation paths.

The preflight passes only if all three environments satisfy all of the
following:

1. All mechanical invariants and protocol/source-manifest checks pass.
2. Candidate `UpperHFPowerAbs <= 0.075^2`.
3. Candidate `LowerLFDriftAbs <= 0.0475^2` and is at least 10% below its paired
   strength-zero value.
4. The normalized joint audit merit is at least 10% below its paired
   strength-zero value.
5. Mean upper-budget feasibility is at least 0.99 and upper-budget violation
   RMS is at most `1e-7`.

The upper gate is absolute rather than a required relative reduction. In the
raw control, upper HPF power can be exactly or nearly zero; a tiny feasible
candidate can then have an arbitrarily bad relative ratio while remaining far
inside the registered physical budget. This boundary is fixed before formal
execution.

## Claim boundary

A pass authorizes a fresh multiseed confirmatory campaign. It does not establish
performance superiority, a domain-general no-tradeoff theorem, or a manuscript
claim by itself. A failure is retained as negative evidence and blocks
multiseed expansion until a new mechanism and protocol version are registered.

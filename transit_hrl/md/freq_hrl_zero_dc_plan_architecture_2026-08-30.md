# Freq-HRL v17 Zero-DC Plan Architecture

## Why v16.2 Was Not Extended

The v16.2 macro-hold gauge was an exact responsibility-coordinate transform,
not a raw-control architecture. It passed all gates in only 2 of 9 development
cells. A follow-up causal smooth-curve prototype reduced synthetic upper HPF8
power but increased lower LPF32 power because a boundary-frozen curve cannot
observe the future actions inside its active macro. Tuning the block EMA or
trend forecast cannot remove that causal delay.

The submission blocker is raw physical separation. It therefore requires a
change to the action parameterization rather than another total-action gauge.

## Upper Macro Plan

At macro boundary `m`, the upper actor samples one target `u_m`. The decoder
executes a cubic smoothstep from the completed previous target to `u_m` over
`K` primitive steps:

```text
s(q) = 3 q^2 - 2 q^3,  q in [0, 1]
u_exec(mK + j) = u_(m-1) + s(j/K) (u_m - u_(m-1)).
```

The target changes only at a macro boundary. Primitive evaluations of the
frozen curve are deterministic decoder outputs, not new upper decisions. The
first target is activated directly to avoid an artificial episode-start ramp.

## Lower Zero-DC Projection

Let `d_j` be the accumulated effective lower residual before primitive step
`j` of a macro, `R = K - j` the remaining step count including the current
step, and `L` the per-coordinate action limit. Future repayment is feasible iff
the post-action debt is within `(R - 1) L`. The nearest feasible action to the
lower proposal `p_j` is therefore

```text
l_j = clip(
  p_j,
  max(-L, -d_j - (R - 1)L),
  min( L, -d_j + (R - 1)L)
).
```

Then `d_(j+1) = d_j + l_j`. At the final step `R = 1`, both feasible bounds are
`-d_j`, so every completed macro satisfies

```text
sum(j=0..K-1) l_j = 0
```

up to floating-point error. The projector exposes the required mean repayment
`-d_j / R` as causal lower-policy context, allowing the learned controller to
avoid late forced corrections.

## Scientific Contract

This architecture is intentionally not function preserving. It changes the
action seen by the plant and must be trained from scratch. No v12, v14.29, or
v16.2 reward result transfers to v17.

Required development evidence:

- raw upper HPF8 respects its frozen absolute budget;
- raw lower LPF32 improves against a capacity-matched direct hierarchy;
- every completed lower macro has zero completion error;
- upper decisions remain one per `K` lower transitions;
- held-out return is noninferior to the direct hierarchy;
- one fixed architecture passes the two-of-three optimizer-seed gate in all
  three MuJoCo environments before any fresh-seed confirmation is dispatched.

The upper smooth decoder and lower zero-DC projector must also be ablated
separately. A positive full-method result without these two controls would not
identify which structural constraint caused the change.

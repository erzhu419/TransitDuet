# MuJoCo v17.9 Prefix-HPF FIR Design

## Failure-Driven Change

V17.8 selected a gain-0.8 FIR because it was the only candidate family member
that met the upper budget on all 120 out-of-fold paths. It recovered only 7/81
failures and destroyed all 32 baseline-feasible Walker2d splits. In contrast,
the gain-one W48 FIR recovered 58 failures and preserved all 32 Walker2d paths,
but exceeded the upper budget on 30 paths.

The failure is not evidence for another global gain sweep. Scaling the complete
upper signal removes its low-frequency component, which must remain assigned to
upper responsibility to keep the complementary lower LPF32 small. V17.9 keeps
gain one and constrains only the quantity named by the upper metric: the causal
HPF8 innovation.

## Causal Projection

At primitive step `t`, the multivariate FIR predicts an upper action from the
current and past total actions. The router first forms the physical interval
that keeps upper and complementary lower actions inside their boxes. Conditional
on the previous seven executed upper actions, the current HPF8 residual is an
affine function of the current upper action. V17.9 projects that residual onto
the remaining prefix energy ball and maps it back to the physical upper action.

This guarantees the upper HPF8 budget at every feasible prefix without scaling
the rolling-mean upper component. It uses no future action, termination time,
reward, disturbance label, or evaluation oracle.

## Frozen Reused-Path Screen

The server-only v17.8 arrays are reused. Candidate widths are 24, 32, 48, and
64; normalized ridge penalties are `1e-5` and `1e-3`; output gain is fixed at
one. Selection remains one shared candidate id across all environments and all
predictions remain leave-one-seed-out.

The full v17.8 gate is inherited: 120/120 valid and upper-feasible paths,
recovery of at least 65/81 failures with environment minima 32 HalfCheetah, 24
Hopper, and 6 Walker2d, preservation of at least 30/32 baseline-feasible
Walker2d paths, and no worse mean lower power in every environment.

Only a complete pass allows the already frozen eight fresh seeds to be
accessed. Passing this reused-path screen would still be mechanism selection,
not a manuscript result.

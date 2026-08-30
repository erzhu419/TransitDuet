# MuJoCo v16.2 Macro-Hold Gauge Outcome

## Decision

`macro_hold_gauge_screen_not_supported`

This is valid development evidence, not confirmatory evidence. All 27
scheduleurm tasks completed on node003 without node binding. The frozen analysis
used nine environment-by-optimizer-seed units, three capacity-matched arms, and
40 held-out paths per arm.

## Frozen Result

- Exact additive-action reconstruction: 9/9 cells.
- Zero router clipping: 9/9 cells.
- Held-out reward noninferiority versus direct reward control: 5/9 cells.
- Absolute upper-HPF8 budget: 4/9 cells.
- At least 10% lower-LPF32 reduction versus the candidate latent split: 6/9 cells.
- At least 10% joint normalized-merit reduction: 6/9 cells.
- All cell gates: 2/9 cells.
- Per-environment two-of-three support gate: 0/3 environments.

Hopper and Walker2d showed useful frequency routing. Their median lower-LF
reductions were 52.6% and 80.4%, respectively. Hopper missed the upper-HF budget
in two cells, while Walker2d missed the reward floor in two cells.

HalfCheetah failed structurally. Its median lower-LF relative reduction was
-1378%, and its median joint-merit relative reduction was -765%. Repeating the
same constant-hold mechanism with more seeds would not address this failure.

## Diagnosis

The constant macro hold conflates a low upper decision rate with a
piecewise-constant upper contribution. Its boundary jumps are visible to the
registered HPF8 audit, while its inability to represent smooth evolution inside
the 16-step macro interval forces slow behavior back into the lower complement.
This explains the combination of upper-HF failures and HalfCheetah lower-LF
failures.

The next mechanism must let one upper decision parameterize a continuous causal
curve over the macro interval. The curve parameters may change only at upper
boundaries; primitive-step evaluation of the frozen curve is not a new upper
decision. Exact additive reconstruction remains mandatory.

## Claim Boundary

Allowed: macro-rate gauge fixing can exactly preserve the executed additive
action, and the constant-hold variant reduced lower-LF leakage in Hopper and
Walker2d development cells.

Forbidden: v16.2 validates cross-environment responsibility separation, reward
improvement, leakage no-tradeoff, fresh-seed confirmation, or a final Freq-HRL
algorithm.

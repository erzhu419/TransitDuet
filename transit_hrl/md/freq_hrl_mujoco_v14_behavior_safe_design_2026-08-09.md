# Freq-HRL MuJoCo v14 Behavior-Safe Development Design

## Evidence status

This document defines a **development** revision. It is not confirmatory
evidence and must not be cited as a positive result.

MuJoCo v13 passed return noninferiority and responsibility-space lower-LF
reduction in HalfCheetah, Hopper, and Walker2d, but failed the global behavioral
gate. HalfCheetah failed raw-lower LF reduction and Hopper failed the absolute
upper-HF budget. The v13 decision and paired tables are immutable.

The failure exposes two objective mismatches in the previous learner:

1. the lower constraint used only the post-transfer responsibility action, so
   a causal bookkeeping transfer could improve `LowerLFDriftAbs` without
   changing `RawLowerLFDriftAbs`;
2. the upper actor had no objective on macro-action discontinuity, although the
   confirmatory endpoint bounded `sqrt(mean(UpperHFPowerAbs))`.

## v14 lower behavioral constraint

Let `l_t` be the raw lower actor action, `r_t` its post-transfer lower
responsibility, and let the two causal EMA states be

```text
z_raw,t  = (1 - alpha) z_raw,t-1  + alpha l_t
z_resp,t = (1 - alpha) z_resp,t-1 + alpha r_t.
```

For an RMS budget `b_l`, define the dimensionless squared excess

```text
e(z; b_l) = max(||z||_RMS / b_l - 1, 0)^2.
```

The v14 joint behavioral cost is

```text
c_lower,t = max(e(z_resp,t; b_l), e(z_raw,t; b_l)).
```

Therefore the cost is zero only when both responsibility attribution and raw
lower behavior satisfy the causal budget. The max aggregation avoids rewarding
one channel for compensating a violation in the other. The existing lower cost
critic and projected primal-dual update consume this scalar cost; the cost
state contains both causal filter states and no future observation.

The responsibility-only branch remains in the development selector as an
ablation, not as the proposed behavior-safe method.

## v14 upper continuity objective

At upper boundary `k`, let `u_k` be the assigned upper responsibility after the
causal transfer and let `b_u` be a transition RMS budget. Define

```text
q_k = max(RMS(u_k - u_k-1) / b_u - 1, 0)^2.
```

Only the first upper credit in macro transition `k` is changed:

```text
R_upper,k = discounted_environment_return_k - beta q_k.
```

The lower reward and the reported episode return remain the unshaped
environment reward. The first macro action of each natural episode has no
transition penalty. The transform is causal because `u_k-1`, the current raw
upper action, and the current transfer state are known at boundary `k`.

Development defaults are `b_u = 0.20` and `beta = 2.0`. They are not
confirmatory thresholds and may be changed only during v14 development. Any
selected value must be frozen before a fresh v15 held-out seed namespace is
created.

## Safety selector

All branches share one initialization, training seeds, checkpoint-selection
seeds, and independent safety-selection paths. The v14 branch registry is:

| Branch | Lower scope | Update | Upper continuity |
|---|---|---|---|
| `no_leakage` | disabled | disabled | disabled |
| `responsibility_guarded_adam_projection` | responsibility | guarded projection | disabled |
| `behavior_guarded_adam_projection` | joint behavior | guarded projection | disabled |
| `behavior_guarded_upper_smooth` | joint behavior | guarded projection | enabled |
| `behavior_scalarized_upper_smooth` | joint behavior | scalarized | enabled |

A constrained branch is eligible only when clustered one-sided bootstrap bounds
on independent safety seeds support all four conditions:

1. environment return is noninferior within the 2% development margin;
2. responsibility `LowerLFDriftAbs` is reduced by at least 10%;
3. `RawLowerLFDriftAbs` is reduced by at least 10%;
4. upper-HF RMS has a one-sided upper bound no greater than 0.10.

Among eligible branches, selection maximizes the minimum normalized safety
slack and then the return lower bound. If no candidate is eligible, the method
falls back to `no_leakage`. Held-out evaluation paths are loaded only after this
selection.

## Required invariants

- Joint lower cost is pointwise no smaller than the responsibility-only cost.
- Responsibility transfer still reconstructs the same nominal executed action.
- Upper shaping changes only upper training credit.
- Lower reward and reported environment return are identical with shaping on or
  off for the same path.
- Every cell reports raw and responsibility online budgets, constraint cost,
  upper transition RMS, continuity penalty, source revision, and source
  manifest.
- v12 and v13 results are analyzed only from their detached frozen worktrees.

## Development and confirmation sequence

1. Run source-bound v14 preflight cells on `node001` through `node006`.
2. Run a development matrix across all three environments and five disturbance
   modes. Inspect selector branch counts, raw/responsibility drift, upper-HF,
   return, saturation, and reconstruction.
3. If v14 is not behavior-safe, modify the algorithm and increment the
   development revision. Do not reinterpret v13 thresholds.
4. Once development is frozen, create a detached source revision and generate
   fresh optimizer, training, checkpoint-selection, safety-selection, and
   held-out seeds for v15.
5. Commit the v15 protocol before dispatch. Analyze all primary endpoints once,
   with family-wise correction and an immutable decision file.

Until step 5 succeeds, the defensible claim remains: v12 supports the
responsibility-transfer claim; v13 rejects the stronger global behavioral
claim; v14 is a mechanism-level repair under development.

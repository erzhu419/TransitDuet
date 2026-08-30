# Freq-HRL Macro-Hold Responsibility Gauge

## Failure Being Repaired

The v16.1 primitive-step adaptive gauge reduced joint normalized frequency
merit relative to the latent split in all nine development cells. It did not,
however, preserve the upper controller's temporal abstraction. The gauge
recomputed the effective upper responsibility on every lower step. Relative to
the latent upper policy, effective upper-HPF power increased by 2.37x to 3.08x
in the three Walker2d cells. A responsibility coordinate assigned to the upper
controller must not silently move at the lower-controller rate.

## Algorithm

`causal_macro_hold_audit_gauge` maintains a primitive-rate causal EMA of the
total additive action. At an explicit upper decision boundary, it copies the
current EMA into a held upper responsibility. During the remainder of the
macro period, that upper coordinate is fixed and the lower responsibility is
the exact additive complement. The EMA cutoff remains adaptive through the
registered normalized HPF8-upper versus LPF32-lower budget imbalance.

At full strength, for total action `a_t = u_t + l_t`, macro index `k(t)`, and
causal low-pass state `m_t`, the split is

```text
U_t = m_tau(k(t))
L_t = a_t - U_t
```

where `tau(k)` is the first primitive step of macro period `k`. Therefore
`U_t + L_t = a_t` at every step, and the split is invariant to additive policy
refactorizations `(u_t + g_t, l_t - g_t)`. It uses no future action or outcome.

## Claim Boundary

The implementation and synthetic tests establish causality, additive
reconstruction, factorization invariance, and macro-rate compatibility. They do
not establish MuJoCo reward or frequency improvements. A frozen development
screen must compare primitive adaptive, macro-hold adaptive, and latent splits
on independent paths before this mode enters a confirmatory protocol.

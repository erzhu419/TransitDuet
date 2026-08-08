# Freq-HRL MuJoCo v10 Causal Responsibility-Transfer Protocol

Date: 2026-08-08

## Evidence Boundary

This protocol is frozen before any v10 training result is inspected. All v5-v10
optimizer, checkpoint-selection, safety-selection, and evaluation paths remain
development data. No v10 path may later be relabeled as confirmatory evidence.

## v9 Outcome And Rejection

The source-bound v9 matrix completed 72/72 cells at revision
`b3cc9c90d615404a75d39af53ad1f216b50c4706`. Every environment-scale row met
the pre-registered reward noninferiority floor, and action clipping was not the
general failure mode. No global upper-action scale passed, however, because at
least one environment at every scale failed the required constrained-branch or
drift-reduction gate. The registered decision is therefore
`no_global_scale_passed`.

This rejects further post-outcome tuning of upper/lower action scales. The v9
protocol requires the next repair to change the causal action decomposition.

## Structural Repair

The additive controller writes

`a_t = clip(u_k + l_t + d_t, -1, 1)`,

where `u_k` is fixed for one upper macro interval, `l_t` is the lower command,
and `d_t` is an external disturbance. In v10, a causal low-frequency estimate
of prior lower commands is transferred only at the next upper boundary:

1. `b_t = (1-alpha) b_(t-1) + alpha l_t`, with `alpha=0.04`;
2. at boundary `k`, request `q_k = b_(k-1)`;
3. clip `q_k` only to the remaining headroom of `u_k`, producing `p_k`;
4. assign `u'_k = u_k + p_k` and `l'_t = l_t - p_k` until the next boundary.

Before final actuator clipping, `u'_k + l'_t = u_k + l_t` exactly. The
transformation therefore changes responsibility attribution without changing
the total nominal action for a fixed pair of raw policy outputs. The estimate
uses no current or future lower action when an upper decision is made. Its
state is included in both policy observations to restore the Markov contract.

The implementation must expose, rather than hide:

- raw and transferred lower low-frequency drift;
- policy-upper, responsibility-upper, raw-lower, and responsibility-lower RMS;
- transfer RMS and headroom saturation rate;
- lower-contribution out-of-unit rate;
- per-step additive reconstruction error before actuator clipping;
- final actuator clipping rate and episode return.

## Fixed Development Matrix

No action-scale or filter sweep is permitted. The fixed configuration is:

- upper action scale: `1.0`;
- lower action scale: `1.0`;
- upper period: `16`;
- causal EMA alpha: `0.04`;
- three registered environments;
- three existing development optimizer replicates;
- four training disturbance modes and all registered evaluation modes.

Two decomposition arms use identical source, seeds, optimizer budgets, and
model capacity:

1. `additive`: the v9 responsibility assignment, augmented only with the
   explicit filter-state observation required for capacity matching;
2. `causal_lf_transfer`: the exact-reconstruction transform above.

Each arm trains `freq_hrl_no_leakage` and `freq_hrl_safe_selector`. The first
comparison isolates the structural transform at equal one-branch compute. The
second compares the complete safe method while disclosing its three-branch
training multiplier.

## Development Gates

The implementation gate requires all unit and integration tests to pass and:

1. future suffix changes cannot alter any prefix transfer output;
2. pre-clipping reconstruction RMS is at most `1e-7` on every path;
3. all filter states reset on natural and budget episode boundaries;
4. source identity and serialized checkpoint hashes verify independently.

The one-branch structural gate compares paired
`causal_lf_transfer/freq_hrl_no_leakage` against
`additive/freq_hrl_no_leakage`. For every environment it requires:

1. mean episode return no worse than 2% below additive;
2. mean responsibility-level `LowerLFDriftAbs` at least 10% below additive;
3. raw-lower drift, transfer saturation, and contribution range are reported;
4. all three optimizer replicates and all held-out disturbance rows exist.

The complete-method gate compares the two safe-selector arms and requires the
same reward and drift conditions. A constrained branch is not mandatory when
the structural transform alone satisfies the responsibility budget; branch
selection and its confidence bounds remain mandatory diagnostics. This is a
new architectural claim, not a reinterpretation of the rejected v9
constraint-only gate.

If the one-branch gate fails, v10 is rejected. If only the complete-method gate
fails, the structural transform may be reported as an ablation result but the
safe complete method cannot advance. Fresh confirmatory seeds remain forbidden
until both gates pass.

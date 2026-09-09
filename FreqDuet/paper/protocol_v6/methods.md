# Methods

## Study question and control setting

FreqDuet studies whether frequency structure in an exogenous demand stream can
be aligned with authority in an asynchronous hierarchical controller. The
upper policy changes an executable target-headway plan at dispatch events; the
lower policy chooses holding at station-arrival events. One upper decision can
therefore span many lower decisions. The current paper controller is
`F_freqduet_protocol_v6_confirmed_main_hiro` under `freqduet-eval-v6`. It is the exact naming alias of the
compact APC/AVL, weight-two controller selected and independently confirmed in
V8.

![Figure 1. Causal frequency-to-authority architecture.](figures/fig1_protocol_v6_method.png)

The simulated bidirectional service operates from
06:00 to 19:00
with a 4-hour clearance period, all
scheduled trips, and a fixed pool of 12
physical vehicles. Passenger arrivals are generated from the local historical
OD-intensity table. The public AFC/APC data are used only for the separate
realism audit and do not calibrate the evaluated policy.

## Causal harmonic demand decomposition

Observed APC arrivals are accumulated in 60-s bins. For bin
`k`, the harmonic basis contains an intercept, a within-day linear trend, and
`K=4` sine/cosine pairs over a
14-h period. Historical OD intensities fit a
ridge-regularized prior for `log(1 + arrival rate)` with ridge
0.01 and initial covariance
0.01. Recursive least squares with forgetting
factor 0.9995 then updates that prior only after the
current observation bin closes. With
basis `phi_k`, coefficients `theta_k`, and pre-update prediction
`lambda_hat_(k|k-1)`, the high-frequency innovation is

```text
r_k = y_k - lambda_hat_(k|k-1).
```

The low-frequency state is the updated nonnegative harmonic rate, its slope,
and its 30-min forecast. The residual and
its exponentially smoothed energy form the high-frequency state. Decisions at
the start of a bin cannot observe arrivals later in that bin. This ordering,
rather than a full-day transform, is the operational no-leakage guarantee.

## Frequency-to-authority allocation

The upper policy receives global low-frequency level, slope and forecast,
together with a scalar high-frequency energy summary and low-frequency OD
structure summaries. The lower policy receives station-direction residual,
residual change, local and global residual energy, the previous holding action,
and compact same-time APC/AVL context: `load`, `capacity`, `queue`, `speed_residual`, `shock_age`, `schedule_slack`, `regularity_hold_target_norm`, `regularity_hold_target_valid`. The scalar energy summary
alerts the upper layer to volatility without giving it the local residual that
drives holding.

The current controller does not use the historical promotion, leakage-penalty,
or legacy causal holding-guard branches. Those mechanisms were development
variants and are not part of the evaluated V6 policy. Consequently, the paper
claim is the behavior of the complete current controller, not an isolated
effect of promotion, leakage regularization, or guard removal.

## Executable upper timetable

The upper ensemble actor produces a headway adjustment bounded to
[-60, 60] s. Every
15 min, the timetable planner maps that action
to an exact terminal headway curve over a
45-min horizon. The
`rolling_zero_sum_delta_v6` projection conserves the cumulative headway
budget over each closed replanning window, preventing hidden phase drift.
Planned launch times are executable: an actual departure occurs no earlier
than both vehicle readiness and the scheduled launch time. Planned and actual
terminal times are logged separately, and terminal shifts are bounded to
[-45, 45] s.

## Discrete lower holding and regularity reward

At each eligible station arrival, the lower categorical ensemble policy
chooses holding from `{0, 5, 10, 15, 20, 30, 45}` s. The sampled action is sent to the
environment without a post-policy projection. The lower state includes the
analytic balancing target derived from a matched predecessor departure and a
same-time AVL estimate of the following vehicle.

Let `g_f` be the pre-action forward departure gap, `g_b` the same-time
follower gap, `h` the executable target headway, and `a` the sampled hold. A
hold predicts gaps `g_f + a` and `max(g_b - a, 0)`. For tolerance
`tau=0.02`, define

```text
q(g,h) = max(|g-h|/h - tau, 0)^2
L(g_f,g_b,h) = 0.5 * [q(g_f,h) + q(g_b,h)].
```

The lower reward receives
`2 * clip(L_before - L_after, -0.25, 0.25)`.
All gaps are frozen before the action. Missing predecessor or follower evidence
adds zero regularity reward and is logged rather than imputed from future
vehicle states.

## Learning architecture

Both levels use off-policy soft actor-critic variants. The upper
`pessimistic_ensemble_sac_v4` uses a pessimistic
10-critic ensemble, discount
0.95, a 64-unit hidden
representation, batch size 64, and
10 updates per episode. The lower
`pessimistic_ensemble_sac_lagrangian_v4` uses a pessimistic
10-critic ensemble, discount
0.99, a 64-unit hidden
representation, batch size 512, and
30 updates per episode. Both learning rates
are `0.0003`; the lower dual learning rate is
`0.0001`. The upper policy begins after a
30-episode lower
warm-up. V8 trains for 40 episodes and evaluates checkpoint 39; V9 trains
without policy or critic freezing for 200 episodes and evaluates checkpoint
199.

## Outcomes

The primary passenger endpoint is restricted total journey time per generated
passenger. Waiting is censored at the evaluation horizon for passengers not
yet boarded; in-vehicle and total journey time are censored at that horizon for
passengers not yet arrived. Secondary outcomes are restricted waiting time,
restricted in-vehicle time, unserved-passenger rate, headway coefficient of
variation, realized vehicle and passenger holding, fleet-denial measures,
terminal execution error, trip completion, and restricted service cost.
Headway CV is the standard deviation divided by the mean over valid recorded
headway events. With restricted waiting `W_R` in minutes, peak fleet `F`, fixed
fleet budget `N`, headway CV `H`, unserved fraction `U`, and trip-completion
fraction `Q`, the secondary scalar is

```text
C_R = W_R / 10
    + max(F - N, 0)^2 / N
    + H
    + 5 U
    + 5 (1 - Q).
```

The evaluated fixed-pool environment makes the overshoot term zero; this
scalar does not directly charge holding or a delayed-readiness denial that is
later retried. It is therefore a secondary summary and does not replace the
passenger and physical outcomes.

## Evaluation and inference

V8 is a preregistered independent 40-episode confirmation with six training
seeds crossed with four untouched evaluation seeds (24 paired rollouts per
policy). V9 is a separately preregistered 200-episode robustness test with
eight new training seeds crossed with eight new evaluation seeds (64 paired
rollouts per policy). The Protocol V6 reference is `F_freqduet_protocol_v6_noguard_hiro`. Both policies
disable the legacy holding guard; the current policy additionally has compact
APC/AVL context and the two-sided regularity objective, so their difference is
a combined-policy contrast.

Uncertainty uses a crossed bootstrap over training and evaluation seeds while
sharing each evaluation-seed resample across paired policies. Two-sided
sign-flip tests operate on training-seed mean differences, with Holm correction
within each metric family. Lower values favor FreqDuet for every reported
outcome. The V8 and V9 estimates are kept separate and are never pooled.

# Methods

## Study question and control setting

FreqDuet studies whether frequency structure in an exogenous demand stream can
be aligned with authority in an asynchronous hierarchical controller. The
upper policy changes an executable target-headway plan at dispatch events; the
lower policy chooses holding at station-arrival events. One upper decision can
therefore span many lower decisions. The current paper controller is the
compact APC/AVL, weight-two configuration that passed the preregistered V8
gate. Its exact identifier and inheritance chain are reported in
Supplementary Section S2.

![Causal frequency-to-authority architecture of the current controller. Historical OD intensities initialize a recursive harmonic demand prior, while online APC arrivals update the filter causally in 60-s bins. Low-frequency level, slope, and forecast features enter the upper policy, which replans an executable terminal headway curve every 15 min over a 45-min horizon under a rolling zero-sum headway budget. Station-local high-frequency innovations and compact same-time APC/AVL context enter the lower policy, which selects from seven discrete holding actions between 0 and 45 s. The two-sided regularity reward uses forward and follower departure gaps frozen before the action. In the current paper configuration, the legacy holding guard, promotion, and leakage penalty are disabled.](figures/fig1_protocol_v6_method.png){#fig:protocol-v6-1}

## Simulation environment and common random numbers

The simulator advances in 1-s steps on one bidirectional corridor with 22
physical stops (two terminals and 20 intermediate stops) and 42 directed
inter-stop links. Each direction is 10.5 km long, comprising 21 links of 500 m.
The input timetable contains 262 trips, split equally between directions.
Service operates from
06:00 to 19:00
with a 4-hour clearance period, all
scheduled trips, a fixed pool of 12 physical vehicles, and
a capacity of 50 passengers per vehicle.

Passenger demand comes from a 20-origin by 14-hour by 20-destination historical
OD-intensity table. Every 20 s, the simulator draws an independent Poisson count
for each active OD cell and assigns each generated passenger a uniform arrival
time within that window. A passenger remains latent until its assigned arrival
time has elapsed. Each service-hour intensity is multiplied by a
`Normal(1, 0.15)` draw clipped to `[0.3, 2.0]`; the historical peak profile is
also shifted by -1, 0, or +1 hour with probabilities 0.2, 0.6, and 0.2. Segment
speed limits update every 300 s by adding Gaussian variation with standard
deviation 1.5 to the corresponding hourly route-history value, clipping the
draw to `[2, 15]`, and applying the segment maximum. An inherited corridor
calibration multiplies reverse-direction OD intensities at X13--X15 by 0.4 for
every policy.

All exogenous draws are supplied by a policy-independent scenario tape keyed by
evaluation seed and process identity. Passenger counts and arrival times,
hourly demand multipliers, peak shifts, and fixed-clock route-speed streams
therefore remain aligned across paired policies even when their action and
learning call sequences differ. V8 and V9 use identical tapes within each
policy pair. The public AFC/APC data are used only for the separate realism
audit and do not calibrate the evaluated policy.

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
basis $\phi_k$, coefficients $\theta_k$, and pre-update prediction
$\widehat{\lambda}_{k\mid k-1}$, the high-frequency innovation is

$$
r_k = y_k - \widehat{\lambda}_{k\mid k-1}.
$$

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
and compact same-time APC/AVL context: load, capacity, queue, speed residual,
shock age, schedule slack, a normalized two-sided holding target, and a
target-valid indicator. The scalar energy summary
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
rolling zero-sum projection conserves the cumulative headway
budget over each closed replanning window, preventing hidden phase drift.
Planned launch times are executable: an actual departure occurs no earlier
than both vehicle readiness and the scheduled launch time. Planned and actual
terminal times are logged separately, and terminal shifts are bounded to
[-45, 45] s.

## Discrete lower holding and regularity reward

At each eligible station arrival, the lower categorical ensemble policy
chooses holding from `{0, 5, 10, 15, 20, 30, 45}` s. Training samples from the policy;
frozen evaluation uses its deterministic output. The chosen action is sent to
the environment without a post-policy projection. The lower state includes the
analytic balancing target derived from a matched predecessor departure and a
same-time AVL estimate of the following vehicle.

Let `g_f` be the pre-action forward departure gap, `g_b` the same-time
follower gap, `h` the executable target headway, and `a` the sampled hold. A
hold predicts gaps `g_f + a` and `max(g_b - a, 0)`. For tolerance
`tau=0.02`, define

$$
\begin{aligned}
q(g,h) &= \left[\max\left(\frac{|g-h|}{h}-\tau,0\right)\right]^2, \\
L(g_f,g_b,h) &= \frac{1}{2}[q(g_f,h)+q(g_b,h)].
\end{aligned}
$$

The lower reward receives
$2\,\operatorname{clip}
(L_{\mathrm{before}}-L_{\mathrm{after}},
-0.25,0.25)$.
All gaps are frozen before the action. Missing predecessor or follower evidence
adds zero regularity reward and is logged rather than imputed from future
vehicle states.

## Learning architecture

Both levels use off-policy soft actor-critic variants
[@haarnoja2018soft]. The upper policy uses a pessimistic
10-critic ensemble, discount
0.95, a 64-unit hidden
representation, batch size 64, and
10 updates per episode. The lower policy
uses a pessimistic constrained
10-critic ensemble, discount
0.99, a 64-unit hidden
representation, batch size 512, and
30 updates per episode. Both learning rates
are `0.0003`; the lower dual learning rate is
`0.0001`. The lower policy's learned constraint
cost is optimized through a Lagrange multiplier, following the constrained-RL
formulation [@miryoosefi2019constraints]. The upper policy begins after a
30-episode lower
warm-up. V8 trains for 40 episodes and evaluates checkpoint 39; V9 trains
without policy or critic freezing for 200 episodes and evaluates checkpoint
199. Both upper and lower actors are deterministic during frozen evaluation,
and the evaluator rejects any change in deployment state across evaluation
seeds.

## External comparators

Three non-learned comparators use the same V6 environment, fixed 12-vehicle
pool, timetable, evaluation seeds, scenario tapes, and exact terminal-release
semantics as the learned controller. `fixed_headway` installs a 360-s terminal
headway in both directions and commands zero intermediate holding.
`rule_holding` uses the same 360-s terminal schedule and applies

$$
a = \operatorname{clip}(360-g_f,0,60),
$$

where `g_f` is the observed forward headway in seconds. `rule_mpc` uses that
same lower holding law and re-evaluates a 60-candidate time-of-day headway grid
at dispatch events. Peak candidates are `{240, 300, 360, 420, 480}` s,
off-peak candidates are `{360, 480, 600, 720}` s, and transition candidates
are `{300, 360, 420}` s. Given a seeded episode-level demand proxy $d$, with
$d\sim\operatorname{clip}(\mathcal{N}(1,0.15),0.3,2.0)$, its
slot-specific surrogate is

$$
\begin{aligned}
H_{\mathrm{ideal}} &= \operatorname{clip}(360/d,240,600), \\
J(H) &= \frac{H}{2}
 + 0.001\max(0,H-H_{\mathrm{ideal}})^2
 + 5\max(0,6000/H-12)^2.
\end{aligned}
$$

The minimizing candidate is converted to an exact launch sequence before
simulation. This rule-MPC is a transparent low-fidelity comparator, not a claim
to represent an optimally tuned or simulator-aware MPC. The fixed-headway
policy is the strong external comparator in this study.

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

$$
\begin{aligned}
C_R &= \frac{W_R}{10}
 + \frac{\max(F-N,0)^2}{N}
 + H
 + 5U
 + 5(1-Q).
\end{aligned}
$$

The evaluated fixed-pool environment makes the overshoot term zero; this
scalar does not directly charge holding or a delayed-readiness denial that is
later retried. It is therefore a secondary summary and does not replace the
passenger and physical outcomes.

## Evaluation and inference

V8 is a preregistered independent 40-episode confirmation with six training
seeds crossed with four untouched evaluation seeds (24 paired rollouts per
policy). V9 is a separately preregistered 200-episode robustness test with
eight new training seeds crossed with eight new evaluation seeds (64 paired
rollouts per policy). The comparator is the no-guard Protocol V6 reference,
whose exact identifier is reported in Supplementary Section S4. Both policies
disable the legacy holding guard; the current policy additionally has compact
APC/AVL context and the two-sided regularity objective, so their difference is
a combined-policy contrast.

Uncertainty uses a crossed bootstrap over training and evaluation seeds while
sharing each evaluation-seed resample across paired policies. Two-sided
sign-flip tests operate on training-seed mean differences, with Holm correction
within each metric family. Lower values favor FreqDuet for every reported
outcome. For each external comparator, its eight evaluation-seed realizations
are crossed with the eight V9 learned training seeds, yielding 64 paired rows;
the crossed analysis retains training seed as the independent policy-training
unit. The V8 and V9 estimates are kept separate and are never pooled.

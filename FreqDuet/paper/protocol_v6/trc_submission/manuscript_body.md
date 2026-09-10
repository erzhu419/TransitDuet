# Introduction

Frequent bus services are vulnerable to self-reinforcing irregularity. A bus
that is delayed encounters more passengers, incurs longer dwell times, and can
fall further behind, while the following bus experiences the opposite feedback.
Holding and dispatch control can interrupt this process, but they exchange
headway regularity for in-vehicle delay, operating time, and fleet availability
[@daganzo2009headway; @xuan2011dynamic]. Comparative studies consequently
quantify trade-offs among holding strategies across regularity and holding
burden [@berrebi2018comparing], and reviews continue to identify
passenger-oriented, real-time, and coordinated control as central research needs
[@ibarra2015planning; @gkiotsalitis2021atstop]. This trade-off makes a strong
fixed-headway policy an essential comparator rather than a nominal baseline.

Reinforcement learning (RL) provides a way to optimize nonlinear bus operations
without solving a new mathematical program at every control event. Existing
work has learned fleet-wide holding policies, represented asynchronous bus
arrivals explicitly, modeled uncertainty with distributional objectives, and
combined holding with other control actions [@wang2020dynamic;
@wang2021asynchronous; @wang2023robust; @rodriguez2023cooperative]. Hierarchical
multi-agent control has also separated strategic action selection from detailed
holding or speed decisions [@yu2024hierarchical]. These studies establish that
learning-based control can coordinate event-driven transit operations, while a
recent systematic evaluation shows that observation design, discrete action
sets, and evaluation configuration can be as consequential as the selected RL
architecture [@xu2025systematic]. A remaining design question is how a learned
hierarchy should allocate exogenous demand information between a slower planner
and a faster station-level controller.

Temporal abstraction supplies one part of the answer. Semi-Markov options and
goal-conditioned hierarchical RL separate decisions by duration or authority
[@sutton1999between; @nachum2018data], and slow representations can improve
high-level subgoal selection [@li2021slow]. Multi-frequency RL instead studies
actions that persist or update at different rates [@lee2020multiple;
@metelli2020persistence; @holt2025evocontrol]. Adjacent policy-learning work
uses Fourier or wavelet representations to expose long trends and local detail
[@huang2025wavelet; @duan2025adaptive]. These approaches motivate multiscale
decision making, but temporal hierarchy, action frequency, and signal
decomposition are not interchangeable. In particular, they do not by themselves
specify which causally observable component of an external demand stream should
inform each control authority.

This study examines that interface in an asynchronous bus-control system. We
propose FreqDuet, which updates a harmonic demand model from completed automatic
passenger counting (APC) bins, passes low-frequency level, slope, and forecast
information to an executable terminal headway planner, and passes station-local
innovations with compact APC and automatic vehicle location (AVL) context to a
discrete holding policy. The upper and lower decisions remain coupled through
the executable target headway, while planned and actual terminal releases are
distinguished under a fixed physical fleet. We make three contributions. First,
we formulate frequency-to-authority allocation as an explicit information
interface for asynchronous hierarchical control. Second, we implement this
interface with causal observation timing, executable timetable semantics, and
coherent sampled holding actions. Third, we evaluate a frozen controller in an
independent short-training confirmation and a separately preregistered
long-training robustness test, retaining the negative robustness result and the
passenger cost of competing with fixed headway.

The empirical claim is deliberately narrower than the architectural proposal.
The confirmatory contrast changes compact APC/AVL context and a two-sided
regularity reward while retaining the harmonic frequency pathway in both arms.
It therefore tests the complete selected controller, not the isolated causal
effect of frequency separation. The study asks whether that controller has a
reproducible operating effect and where the effect stops, rather than treating
all favorable metrics or development variants as confirmation.

# Related Work

## Timetable, dispatch, and holding control

Classical bus-control research treats instability through schedules, dispatch
rules, holding, speed control, or combinations of these interventions. Dynamic
headway control can reduce bunching with less schedule slack than conventional
schedule adherence [@daganzo2009headway], whereas virtual-schedule control can
jointly target schedule reliability and regular headways [@xuan2011dynamic].
Prediction-based holding can improve the compromise between regularity and
holding time, but its performance depends on prediction quality and operating
configuration [@berrebi2018comparing]. Broader reviews organize these methods by
planning horizon, control action, objective, and real-time feasibility
[@ibarra2015planning; @gkiotsalitis2021atstop]. The common operational lesson is
that regularity is not free: holding can protect downstream headways while
delaying onboard passengers and occupying vehicles.

FreqDuet retains this physical accounting. Its upper action changes an
executable terminal headway plan, and its lower action delays a vehicle at an
intermediate stop. These are different control locations, but both alter future
departure gaps. The evaluation therefore reports passenger journey, realized
holding, delayed fleet readiness, launch completion, and headway variation
separately. This differs from interpreting a regularity-weighted scalar as a
complete measure of passenger or operator performance.

## Learning-based control for asynchronous bus operations

Deep RL has expanded bus holding from local rules to coordinated policies.
Fleet-level multi-agent RL captures interactions that are difficult to encode in
a fixed local law [@wang2020dynamic]. Event-driven formulations address the fact
that buses reach control stops asynchronously and that other agents may act
between two decisions of a focal bus [@wang2021asynchronous]. Distributional and
meta-learning extensions target demand surges, traffic perturbations, and
service interruptions [@wang2023robust], while cooperative formulations combine
holding with stop skipping and explicitly model nearby agents
[@rodriguez2023cooperative]. Hierarchical multi-agent RL has separated a
high-level choice of intervention from a lower-level action magnitude
[@yu2024hierarchical].

This literature also cautions against equating a larger model with a stronger
controller. A systematic bus-bunching evaluation found that compact spacing
states can outperform more elaborate states and that discrete holding can match
continuous actions while being easier to implement [@xu2025systematic]. FreqDuet
accordingly uses event-specific states and a small discrete holding alphabet.
Its distinction is not a new generic multi-agent critic. Instead, it defines an
information boundary between a dispatch-scale planner and an arrival-scale
holding policy, then evaluates the resulting controller under shared scenario
tapes and fixed fleet semantics.

## Hierarchical and frequency-aware decision making

Hierarchical RL traditionally separates behavior through temporally extended
options, learned goals, or subgoal representations [@sutton1999between;
@nachum2018data]. Slow-feature objectives make this temporal intuition explicit
by learning high-level representations that change slowly enough to support
abstract exploration [@li2021slow]. These methods concern the organization of
decisions or endogenous state representations; they do not necessarily split an
external time series into signals with different control ownership.

A second line of work models control rates directly. Action-persistence methods
adapt how long an action is repeated [@metelli2020persistence], multi-frequency
RL handles action variables that update at different periods
[@lee2020multiple], and bi-level high-frequency control learns slow and fast
policies jointly [@holt2025evocontrol]. Their central object is the rate at which
actions are selected. In FreqDuet, asynchronous action rates are already fixed
by transit events; the additional question is which part of observed demand is
available to each policy.

A third line applies spectral representations to policy inputs, context, or
action sequences. Learnable wavelet policies decompose long-horizon sequences
into coarse and fine components [@huang2025wavelet], and low-frequency
truncation has been used to summarize long context for multi-agent RL
[@duan2025adaptive]. FreqDuet adopts the multiscale motivation but uses a causal
harmonic estimator with a historical prior rather than a full-trajectory
transform. Low-frequency demand summaries inform upper planning; station-local
pre-update innovations inform lower holding. This allocation is a control
contract, not merely a richer shared representation. The current experiment,
however, compares complete policies and does not isolate this contract from the
simultaneous APC/AVL and reward changes.

## Passenger data and realism

Automated fare collection (AFC), APC, and AVL systems provide complementary
views of passenger demand and vehicle motion. Smart-card records support
strategic, tactical, and operational analysis, but their fields and sampling
processes vary by agency [@pelletier2011smart]. FreqDuet uses local historical OD
intensities to initialize its demand prior. Public MTA AFC and Halifax APC
subsets are used only to compare normalized demand shapes; they do not share the
evaluated network, service day, or measurement process. The external-data
analysis is therefore a realism audit, not policy validation on those agencies.

# Methods

## Study question and control setting

FreqDuet studies whether frequency structure in an exogenous demand stream can
be aligned with authority in an asynchronous hierarchical controller. The
upper policy changes an executable target-headway plan at dispatch events; the
lower policy chooses holding at station-arrival events. One upper decision can
therefore span many lower decisions. The current paper controller is
`F_freqduet_protocol_v6_confirmed_main_hiro` under `freqduet-eval-v6`. It is the exact naming alias of the
compact APC/AVL, weight-two controller that passed the preregistered V8 gate.

![Causal frequency-to-authority architecture of the current controller. Historical OD intensities initialize a recursive harmonic demand prior, while online APC arrivals update the filter causally in 60-s bins. Low-frequency level, slope, and forecast features enter the upper policy, which replans an executable terminal headway curve every 15 min over a 45-min horizon under a rolling zero-sum headway budget. Station-local high-frequency innovations and compact same-time APC/AVL context enter the lower policy, which selects from seven discrete holding actions between 0 and 45 s. The two-sided regularity reward uses forward and follower departure gaps frozen before the action. In the current paper configuration, the legacy holding guard, promotion, and leakage penalty are disabled.](fig1_protocol_v6_method.pdf){#fig:protocol-v6-1}

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
chooses holding from `{0, 5, 10, 15, 20, 30, 45}` s. Training samples from the policy;
frozen evaluation uses its deterministic output. The chosen action is sent to
the environment without a post-policy projection. The lower state includes the
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

Both levels use off-policy soft actor-critic variants
[@haarnoja2018soft]. The upper
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

```text
a = clip(360 - g_f, 0, 60),
```

where `g_f` is the observed forward headway in seconds. `rule_mpc` uses that
same lower holding law and re-evaluates a 60-candidate time-of-day headway grid
at dispatch events. Peak candidates are `{240, 300, 360, 420, 480}` s,
off-peak candidates are `{360, 480, 600, 720}` s, and transition candidates
are `{300, 360, 420}` s. Given a seeded episode-level demand proxy `d`, with
`d ~ clip(Normal(1, 0.15), 0.3, 2.0)`, its slot-specific surrogate is

```text
H_ideal = clip(360 / d, 240, 600)
J(H) = H / 2 + 0.001 max(0, H - H_ideal)^2
       + 5 max(0, 6000 / H - 12)^2.
```

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
outcome. For each external comparator, its eight evaluation-seed realizations
are crossed with the eight V9 learned training seeds, yielding 64 paired rows;
the crossed analysis retains training seed as the independent policy-training
unit. The V8 and V9 estimates are kept separate and are never pooled.

# Results

## Independent confirmation at 40 episodes

V8 passed the registered effect/no-harm gate for the complete current policy
(Fig. 2; Table 1). Relative to the Protocol V6 reference, headway CV changed by
-0.022 [-0.038, -0.008]. Its
crossed-bootstrap interval excluded zero, whereas the Holm-adjusted
training-seed sign-flip result was
$p=0.125$. Restricted
passenger journey changed by
-0.263 [-0.837, +0.175]
min. The latter interval crossed zero but satisfied the preregistered journey
no-harm margin. Thus V8 is gate-positive under its preregistered criteria; it
is not a familywise-significant effect at 0.05 and does not establish a
passenger-journey benefit.

![Independent confirmation and long-training robustness. Points show paired mean differences between the current policy and the Protocol V6 reference config named `F_freqduet_protocol_v6_noguard_hiro`; bars show 95% crossed-bootstrap confidence intervals over training and evaluation seeds. Both configurations disable the legacy causal holding guard. The current policy additionally uses compact APC/AVL context and the two-sided departure-regularity objective, so this is a combined-policy comparison rather than an isolated guard effect. Lower values favor the current policy. V8 contains 24 paired rollouts (six training seeds by four untouched evaluation seeds) and passed the registered effect/no-harm gate. Its Holm-adjusted training-seed sign-flip result was p=0.125, so the figure labels V8 as gate-positive rather than familywise significant. V9 contains 64 paired rollouts (eight by eight); its passenger-journey interval favored FreqDuet, but the headway-CV effect did not meet the registered magnitude and interval gate, so V9 is reported as not confirmed.](fig2_protocol_v6_confirmation_robustness.pdf){#fig:protocol-v6-2}

**Table 1. Current policy minus the Protocol V6 reference.** Values are paired
mean differences with crossed-bootstrap 95% confidence intervals. Lower is
better. Holm-adjusted sign-flip p-values use training-seed mean differences.

| Outcome | V8 delta [95% CI] | V8 Holm p | V9 delta [95% CI] | V9 Holm p |
| --- | --- | --- | --- | --- |
| Restricted passenger journey (min) | -0.263 [-0.837, +0.175] | 0.938 | -1.242 [-2.204, -0.536] | 0.047 |
| Restricted passenger wait (min) | -0.185 [-0.470, +0.050] | 0.375 | -1.011 [-1.878, -0.397] | 0.047 |
| Restricted in-vehicle time (min) | -0.078 [-0.376, +0.147] | 1.000 | -0.232 [-0.355, -0.121] | 0.047 |
| Headway coefficient of variation | -0.022 [-0.038, -0.008] | 0.125 | -0.009 [-0.028, +0.006] | 0.031 |
| Unserved passengers (percentage points) | +0.02 [-0.00, +0.11] | 1.000 | +0.00 [+0.00, +0.02] | 0.250 |
| Realized holding (s/launched trip) | -1.7 [-41.8, +29.4] | 1.000 | -37.6 [-59.8, -16.2] | 0.070 |
| Trips denied at least once (percentage points) | +0.48 [-9.21, +8.81] | 1.000 | -1.65 [-3.96, -0.22] | 0.117 |
| Restricted service cost | -0.040 [-0.079, -0.007] | 0.125 | -0.110 [-0.207, -0.042] | 0.047 |

## Long-training robustness was not confirmed

At 200 episodes, the same policy improved restricted journey by
-1.242 [-2.204, -0.536]
min relative to the reference. The headway-CV difference was
-0.009 [-0.028, +0.006]. Seven of
eight training-seed CV differences were negative, but the interval included
zero and the mean improvement did not reach the registered 0.02 threshold.
V9 therefore returned `longtrain_not_confirmed`; the favorable journey result
cannot be relabelled as confirmation of the registered regularity effect.

## External baselines reveal a passenger-regularity trade-off

The source-identical V9 comparison (Fig. 3; Table 2) shows that FreqDuet had
lower headway CV than fixed headway,
-0.219 [-0.238, -0.198], and lower
restricted service cost,
-0.122 [-0.180, -0.050].
However, restricted journey was higher by
+2.470 [+1.874, +3.188]
min. FreqDuet also used
+286.8 [+269.7, +302.0] more
holding seconds per launched trip and had a
+64.10 [+61.97, +66.03]
percentage-point higher denied-trip rate. All trips were eventually launched
and completed in the aggregated learned-policy results, so this denial measure
captures delayed fleet readiness rather than permanent trip cancellation.
Because the restricted service-cost scalar does not directly charge holding or
retried readiness denials, its favorable difference cannot be interpreted as
passenger-time or fleet-readiness superiority.

FreqDuet reduced restricted journey relative to rule holding by
-3.224 [-4.727, -1.885]
min and relative to rule MPC by
-26.511 [-31.003, -21.628]
min. The supported external conclusion is therefore narrower than universal
superiority: FreqDuet reduced restricted journey relative to the two rules and
produced more regular service than fixed headway, while fixed headway remained
better for passenger journey and fleet-readiness burden.

![External baseline trade-off under the V9 source contract. Points show paired mean differences between FreqDuet and each external baseline; bars show 95% crossed-bootstrap confidence intervals over eight training and eight evaluation seeds (64 paired rollouts). Lower values favor FreqDuet. FreqDuet improved regularity and restricted service cost relative to fixed headway but increased passenger journey time. It reduced passenger journey time relative to rule holding and rule MPC. Exact two-sided sign-flip tests and Holm-adjusted values are provided in the source table and are not encoded as significance symbols in the figure.](fig3_protocol_v6_external_tradeoff.pdf){#fig:protocol-v6-3}

**Table 2. FreqDuet minus external baseline under V9.** Values are paired mean
differences with crossed-bootstrap 95% confidence intervals. Lower is better.
The complete outcome and adjusted-test table is in the Supplementary Material.

| Baseline | Journey min | Headway CV | Denied trips (pp) | Service cost |
| --- | --- | --- | --- | --- |
| Fixed headway | +2.470 [+1.874, +3.188] | -0.219 [-0.238, -0.198] | +64.10 [+61.97, +66.03] | -0.122 [-0.180, -0.050] |
| Rule holding | -3.224 [-4.727, -1.885] | -0.073 [-0.082, -0.062] | -3.17 [-5.42, -1.90] | -0.351 [-0.496, -0.221] |
| Rule MPC | -26.511 [-31.003, -21.628] | +0.006 [-0.006, +0.020] | +61.52 [+59.09, +63.77] | -2.541 [-2.984, -2.060] |

## Physical execution audit

Relative to the Protocol V6 reference, V9 reduced realized holding by
-37.6 [-59.8, -16.2] s
per launched trip and the denied-trip rate by
-1.65 [-3.96, -0.22]
percentage points (Fig. 4). These are full-policy differences, not isolated
effects of the regularity reward. Against fixed headway, however, the current
policy used
+286.8 [+269.7, +302.0] more
holding seconds per launched trip and increased the denied-trip rate by
+64.10 [+61.97, +66.03]
percentage points. All trips were eventually launched and completed in the
aggregated learned-policy results, so denial records delayed fleet readiness
rather than permanent trip cancellation.

![Paired physical outcomes of the current policy. Points show mean paired differences between the full current policy and the Protocol V6 reference configuration; bars show 95% crossed-bootstrap confidence intervals. Negative values favor the current policy. V8 contains 24 paired rollouts and V9 contains 64. The current policy differs from the reference by both compact APC/AVL context and the two-sided departure-regularity objective, so these panels describe the combined policy's physical behavior rather than an isolated legacy-guard effect.](fig4_protocol_v6_physical_outcomes.pdf){#fig:protocol-v6-4}

## External data support demand-shape realism only

The FreqDuet OD input has a morning peak similar in timing to the bounded MTA
AFC subset, while its normalized hourly profile differs from the Halifax APC
subset (Fig. 5). The balanced audit contains 39 complete MTA station-complex
days (936 rows) and seven complete Halifax routes across 37 route-days (979
rows). Because systems, dates, sampling units, and measurement processes are
unmatched, these comparisons are descriptive checks of demand-shape
plausibility. They are not same-day calibration, route-family policy tests, or
field-effect estimates.

![External passenger-count demand-shape audit. Panel a compares separately normalized hourly demand shapes from the FreqDuet OD input, a complete-day subset of the bounded public MTA AFC cache (936 source rows; 39 station-complex days), and a complete-route subset of the bounded public Halifax APC cache (979 source rows; 7 routes and 37 route-days). Panel b summarizes the corresponding demand-period shares; the FreqDuet input contains 20 origin series. The balanced-cache derivation excludes incomplete pagination fragments. This remains a descriptive audit across unmatched systems and dates, not a population estimate, same-day calibration, field-policy evaluation, or evidence of deployed control benefit.](fig5_protocol_v6_external_realism.pdf){#fig:protocol-v6-5}

# Discussion

## Principal findings

FreqDuet produced a positive result under its registered V8 gate, but the
evidential scope is bounded. Under the independent V8 protocol, the frozen
complete controller met the registered short-training
regularity gate and the passenger-journey no-harm condition. The confidence
interval for the paired headway-CV difference excluded zero, although the
Holm-adjusted training-seed sign-flip result was $p=0.125$. With six training
seeds, the sign-flip randomization distribution is coarse. Under the separate
V9 protocol, passenger journey improved relative to the same Protocol V6
reference (Holm-adjusted $p=0.047$), but the registered long-training
regularity magnitude was not confirmed. These stages answer different
questions and are not pooled: V8 is gate-positive under its registered
short-training criteria, whereas V9 did not recover its
registered regularity magnitude under the longer learning horizon.

The V9 headway sign-flip result was Holm-adjusted $p=0.031$ even though the
crossed-bootstrap interval included zero. These statistics use different
sampling units: the sign-flip test reduces each trained policy to its mean over
evaluation seeds, whereas the crossed bootstrap propagates variation from both
training and evaluation seeds. The registered V9 gate required the latter
effect-size interval and was not passed; the smaller sign-flip value is not used
to relabel V9 as a confirmation.

This pattern matters for learning-based bus control. Prior work demonstrates
that RL can coordinate asynchronous holding and adapt to operational uncertainty
[@wang2020dynamic; @wang2021asynchronous; @wang2023robust], but the present
results show why training duration belongs in the evaluation contract. A policy
can retain a favorable passenger outcome while the regularity effect used for
selection weakens. Reporting only the selected checkpoint or pooling training
horizons would conceal that failure mode.

## Passenger service versus regularity

The fixed-headway comparison exposes a second boundary. FreqDuet generated
substantially more regular headways and a lower restricted service-cost scalar,
but fixed headway delivered lower restricted passenger journey and imposed far
less holding and delayed fleet readiness. This is consistent with the classical
observation that holding trades regularity against onboard delay and vehicle
productivity [@daganzo2009headway; @xuan2011dynamic;
@berrebi2018comparing]. More regular service is therefore not sufficient to
claim better passenger service.

The scalar service cost should also not be used to reverse this conclusion. In
the evaluated fixed-pool environment its fleet-overshoot term is identically
zero, and it does not directly price realized holding or a readiness denial that
is later retried. The separate physical outcomes reveal burdens hidden by that
summary. A passenger-centered deployment objective would need to price holding,
readiness delay, and censored journey outcomes explicitly rather than relying on
headway variation as a surrogate.

The two rule baselines bound narrower questions. The proportional controller is
a transparent local holding law, and the rule-MPC uses a coarse time-of-day
surrogate rather than a calibrated prediction model. Improvement over those
rows does not establish superiority to the broader holding and model-predictive
control literature. The fixed-headway row is the demanding external comparison
for this experiment.

## Interpretation of the frequency interface

The architecture gives low-frequency demand summaries to the upper planner and
station-local innovations to the lower controller while preserving causal
observation timing. This is a concrete extension of temporal abstraction: the
hierarchy separates not only when policies act, but also which component of an
external process each policy can observe. The distinction complements work on
temporally extended goals [@sutton1999between; @nachum2018data], multiple action
frequencies [@lee2020multiple; @metelli2020persistence;
@holt2025evocontrol], and frequency-domain representations
[@huang2025wavelet; @duan2025adaptive].

The experiments do not establish that this interface caused the V8 result. The
current and reference controllers both retain the harmonic path, while the
current controller also changes compact APC/AVL context and the two-sided
regularity reward. Plausible explanations therefore include better local
observability, denser action-aligned credit, their interaction with the
frequency state, or ordinary training variation. The supported conclusion is
about the complete controller. An isolated frequency claim requires a fresh,
same-stage comparison against no-frequency, raw-history, all-frequency, and
swapped-authority controls under the current physical semantics.

## External realism and applicability

The MTA AFC and Halifax APC profiles show that the simulator demand is not a
featureless stationary input and that its morning structure is plausible in at
least one external comparison. They also show a mismatch with the separately
normalized Halifax profile. That mismatch is informative because AFC entries,
APC boardings, OD intensities, and vehicle locations are different measurement
objects [@pelletier2011smart]. The audit supports demand-shape plausibility only.
It does not demonstrate transfer to another route or agency.

The implemented controller is most applicable to high-frequency services with
terminal dispatch control, event-time AVL, station-level passenger observations,
and a known physical fleet. Its causal filter can operate online because it
updates only after a count bin closes. The current evidence does not establish
performance with missing APC streams, multi-line vehicle sharing, driver
noncompliance, or field dispatch interventions.

## Limitations

Five limitations define the present claim. First, V9 did not confirm the
registered 200-episode regularity effect; long-horizon learning stability remains
unresolved. Second, the confirmatory contrast is bundled and lacks current-stage
NoFreq, RawHistory, AllFreq, and swapped-authority controls, so frequency
allocation is not causally isolated. Third, the strong fixed-headway baseline is
better on passenger journey and fleet-readiness burden, despite worse headway
CV. Fourth, the policy is evaluated in one simulated network initialized from a
local historical OD table. The public external data are unmatched descriptive
profiles, not same-day AFC/APC/AVL calibration, route-family policy tests, or a
field treatment effect. Fifth, the inherited simulator scales reverse-direction
OD intensities at X13--X15 by 0.4. That calibration is common to all paired
policies but was not re-estimated from the external data, further limiting
transportability beyond the evaluated corridor.

## Implications for further evaluation

The next decisive experiment is not another gate tuned on V8 or V9. It is a
new, frozen factorial confirmation that holds APC/AVL context, reward, physical
fleet, and training protocol constant while varying only frequency allocation.
Such a study should use unseen route and service-day families and retain
fixed-headway passenger journey and fleet burden as co-primary constraints.
Separately, replacing the current restricted scalar with an objective that
prices holding and readiness delay would test whether the regularity gain can be
converted into passenger benefit. Until those experiments are complete,
FreqDuet should be read as a causally implemented and independently evaluated
frequency-to-authority design with one registered gate-positive short-training
signal, not as a universally superior bus controller.

# Conclusions

FreqDuet implements a causal frequency-to-authority interface for asynchronous
bus control: historical and completed-bin demand information defines a smooth
state for executable terminal headway planning, while station-local innovations
and same-time APC/AVL context inform discrete intermediate-stop holding. The
frozen controller met its independent short-training headway-regularity gate
without violating the registered passenger-journey no-harm condition. In the
separate long-training test, passenger journey improved relative to the Protocol
V6 reference, but the registered regularity magnitude was not confirmed.

The external comparison clarifies the practical meaning of these results.
FreqDuet was more regular than fixed headway and improved passenger journey over
the weaker rule-holding and rule-MPC baselines, yet fixed headway remained better
for passenger journey, holding burden, and fleet readiness. The study therefore
supports a registered gate-positive short-horizon result for the complete
selected controller. Because the Holm-adjusted sign-flip result was $p=0.125$,
it does not establish a familywise-significant effect, long-run robustness, or
universal superiority. A fresh factorial confirmation and matched route-day
evaluation are required to isolate frequency allocation and determine whether
its regularity benefit can be converted into a passenger-service benefit.

# Data and Code Availability

The repository contains the small immutable CSV and JSON artifacts supporting
the V8 confirmation, V9 robustness test, and V9 external-baseline comparison,
together with exact seed contracts, resolved configurations, source commits,
derived figure data, and assembly scripts. Checkpoints and full training logs
remain on the HPC filesystem and are not part of the manuscript bundle. A
persistent archival identifier will be added after the manuscript scope and
release snapshot are frozen.

The external realism audit uses bounded subsets derived from public MTA AFC and
Halifax Transit APC sources. Source endpoints, selection rules, coverage, and
derived hourly profiles are recorded in the packaged data manifests. MTA Bus
Time route, stop, and AVL records are stored in a separate offline cache; the API
credential used for download is neither stored in the repository nor required
to rebuild the manuscript from the derived data. These public data support only
the descriptive demand-shape audit. They are not same-day AFC/APC/AVL
calibration or observed field outcomes for the simulated network.

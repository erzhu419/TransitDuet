# Freq-HRL: Auditing and Guarded Restoration of Frequency Responsibility in Hierarchical Reinforcement Learning

Authoritative venue-neutral draft, 2026-08-30

Evidence source: `transit_hrl/evidence/authoritative_registry_v1.json`

## Abstract

Hierarchical reinforcement learning (HRL) separates decisions across temporal
scales, but different decision periods do not ensure that the levels assume
different frequency responsibilities. A lower policy may learn persistent
low-frequency corrections, while an upper policy may oscillate at macro-step
boundaries. We introduce Freq-HRL, a two-timescale actor-critic protocol that
audits this failure in action-effect space and applies a guarded post-training
restoration transaction. The protocol distinguishes raw lower-action drift from
responsibility-space drift, preserves a paired reward floor, and abstains when
no candidate satisfies frozen design-fold gates. In a preregistered MuJoCo
study, an earlier Freq-HRL configuration reduced responsibility-space lower-
frequency drift by 78.2--89.6% across HalfCheetah-v5, Hopper-v5, and Walker2d-v5
while meeting return noninferiority (24 optimizer replicates per arm and task).
A stricter follow-up did not support universal raw behavioral separation:
HalfCheetah missed the raw lower-frequency gate and Hopper missed the upper
high-frequency gate. We then evaluated a frozen restoration portfolio on 48
fresh trained anchors. It passed for 16/16 HalfCheetah, 16/16 Hopper, and 15/16
Walker optimizer seeds, with two-sided 95% Wilson lower bounds of 0.806, 0.806,
and 0.717. Thirty-eight selections were function-preserving responsibility
routers, nine were guarded actor updates, and one abstained; all 47 accepted
cells respected the held-out reward floor. A separate synthetic time-series
control study produced mixed results: 8 of 12 Holm-controlled contrasts favored
Freq-HRL, one showed significant harm, and three were inconclusive. The evidence
therefore supports an auditable responsibility-restoration protocol, not reward
dominance or universal physical frequency separation.

## 1. Introduction

Temporal abstraction is a central motivation for hierarchical reinforcement
learning. Options formalize temporally extended actions [@sutton1999between],
option-critic learns their internal policies and termination conditions
[@bacon2017option], and modern continuous-control methods learn upper goals and
lower goal-conditioned behavior [@nachum2018data]. These constructions decide
*when* each level acts. They do not, by themselves, determine *which temporal
frequencies of the resulting control effect* each level is responsible for.

This distinction matters in environments driven by exogenous time series. A
slowly acting upper policy can still induce high-frequency action changes at
macro boundaries. Conversely, a lower policy that acts every step can accumulate
a persistent bias and become the de facto long-horizon planner. The resulting
policy may achieve high return while violating the intended division of labor.
The issue is not solved by adding Fourier or wavelet features: spectral
representations can expose temporal scales [@konidaris2011fourier;
@wang2026wavelet], but they do not identify which policy caused an action effect.

Freq-HRL treats frequency separation as an explicit, auditable responsibility
contract. It combines a semi-Markov upper/lower actor-critic with causal filters,
domain-specific action-effect operators, separate raw and responsibility-space
diagnostics, and a reward-constrained restoration stage. The restoration stage
does not force every anchor through an update. It selects from a frozen set of
function-preserving responsibility routers and guarded actor transactions, or
abstains if all candidates fail. This makes the method closer to constrained
policy improvement [@achiam2017cpo] than to an unconstrained auxiliary loss,
while retaining an empirical, path-conditional guarantee rather than claiming a
global safety theorem.

An additive hierarchy also has a structural identification problem. If the
environment sees only the sum of upper and lower effects, transferring any
causal signal from one level to the other leaves behavior unchanged. We make
this gauge freedom explicit and define a causal gauge-fixed responsibility
coordinate from the total action. This separates what can be identified from
behavior from what requires an architectural convention.

Our contributions are:

1. We define and implement a causal distinction between raw lower-frequency
   behavior, responsibility-space lower-frequency drift, and upper-policy
   high-frequency power. This prevents a representation result from being
   reported as a behavioral result.
2. We formalize the non-identifiability of raw additive upper/lower
   factorizations and provide a causal, exactly reconstructing gauge-fixed
   responsibility layer whose full-strength output depends only on total action.
3. We introduce a domain-neutral guarded restoration portfolio with disjoint
   design and validation paths, fold-wise eligibility, a paired reward floor,
   exact trace-invariance requirements for function-preserving candidates, and
   counted abstention.
4. We report preregistered fresh-seed evidence together with the negative and
   mixed results that bound the claim: responsibility restoration is reliable on
   the frozen MuJoCo protocol, raw separation is not universal, and return does
   not uniformly exceed matched learned baselines.

The paper's central claim is deliberately narrower than the original project
vision: Freq-HRL provides a reproducible way to audit and restore frequency
responsibility under a reward floor. The present evidence does not establish a
domain-general performance advantage.

## 2. Related Work

### 2.1 Temporal abstraction and hierarchical control

The options framework connects temporally extended actions to semi-Markov
decision processes [@sutton1999between]. Option-critic learns options end to end
[@bacon2017option], while HIRO addresses off-policy correction between learned
upper and lower policies [@nachum2018data]. Controlled-effect HRL shifts the
hierarchical interface from primitive actions to environment transformations
[@corcoll2022disentangling]. Freq-HRL shares the focus on effects, but asks a
different question: whether slow and fast control effects are assigned to the
intended levels and whether that assignment can be restored without violating a
paired reward floor.

### 2.2 Frequency-aware representation

Fourier bases provide fixed spectral features for value approximation
[@konidaris2011fourier]. Recent wavelet predictive representations model
multi-scale changes in non-stationary MDP sequences [@wang2026wavelet]. Freq-HRL
uses causal slow and fast state streams, but its primary object is not a spectral
state embedding. It measures the frequency content of policy action effects and
separates physical behavior from responsibility coordinates.

### 2.3 Constrained actor-critic updates

PPO provides the on-policy actor-critic foundation used by the shared Freq-HRL
core [@schulman2017ppo]. CPO formalizes policy optimization under auxiliary
constraints [@achiam2017cpo]. SAC and TD3 are included as off-policy continuous-
control comparators in the time-series study [@haarnoja2018sac;
@fujimoto2018td3]. Freq-HRL's restoration gate is empirical and paired: it
requires zero observed reward-floor violations on frozen paths. It must not be
interpreted as CPO's analytical near-constraint guarantee.

## 3. Frequency Responsibility

### 3.1 Two-timescale control

Let an environment evolve at primitive time (t), with an upper decision every
(K) primitive steps. At macro index (m=\lfloor t/K\rfloor), the upper policy
samples (u_m \sim \pi_U(\cdot\mid s^U_m)); the lower policy samples
(l_t \sim \pi_L(\cdot\mid s^L_t,u_m)). The domain adapter maps these decisions
to an executed action (a_t). Freq-HRL stores one semi-Markov upper transition
per macro interval and one lower transition per primitive step, and applies
independent PPO likelihood ratios to the two policies.

The state adapter is causal. A slow exponential state and a faster exponential
state are updated using observations available through time (t). Their
difference and residual expose middle- and high-frequency innovations without
future leakage. Alternative causal Fourier, state-space, neural state-space,
Poisson harmonic, and wavelet encoders exist in the implementation, but the
confirmatory MuJoCo claim does not depend on an encoder-comparison result.

### 3.2 Action-effect diagnostics

Raw actions are not comparable across domains: holding time, inventory change,
and motor torque have different accumulated effects. A domain action-effect
operator therefore maps upper and lower action histories to aligned effect
sequences (e^U_{1:T}) and (e^L_{1:T}). Let (L_w\) be the registered causal
low-pass operator and (H_w=I-L_w\) its residual. The raw diagnostics are

\[
D^{\mathrm{raw}}_L = T^{-1}\lVert L_w(e^L)\rVert_2^2,
\qquad
P^{\mathrm{raw}}_U = T^{-1}\lVert H_w(e^U)\rVert_2^2.
\]

A causal responsibility operator additionally constructs (r^U_t,r^L_t) so
that their sum reconstructs the registered total effect up to numerical error.
The responsibility diagnostic is

\[
D^{\mathrm{resp}}_L = T^{-1}\lVert L_w(r^L)\rVert_2^2.
\]

The distinction is substantive. A responsibility router can lower
(D^{\mathrm{resp}}_L) while preserving every executed action. Such a transaction
repairs attribution, not physical behavior. Raw separation requires improvement
in (D^{\mathrm{raw}}_L) and the upper high-frequency budget as separate gates.

### 3.3 Additive gauge non-identifiability

Assume the domain adapter and transition kernel depend on the hierarchy through
the additive total effect (a_t=e^U_t+e^L_t). For any causal transfer (g_t) that
keeps the component actions feasible, define

\[
e^{U\prime}_t=e^U_t+g_t,\qquad
e^{L\prime}_t=e^L_t-g_t.
\]

**Proposition 1 (raw factorization is not behaviorally identifiable).** Under
shared initial state and environment randomness, the original and transformed
hierarchies induce identical state, executed-action, and reward trajectories.
Consequently, an objective or diagnostic that observes only environment
trajectories cannot identify the raw upper/lower factorization.

*Proof.* At every (t), the transformed total equals the original total. The
domain adapter therefore emits the same executed action. Induction through the
shared transition randomness gives the same next state and reward, completing
the trajectory-wise argument. (\square)

This proposition explains why responsibility restoration and raw policy
distillation are different estimands. Successful responsibility routing cannot
by itself imply improvement in either raw diagnostic.

### 3.4 Causal gauge fixing

Let (P) be any deterministic causal operator on the total action history; the
implementation uses a causal exponential low pass followed, when configured,
by a lower-component feasibility projection. Define

\[
r^U_t=P(a_{1:t})_t,\qquad r^L_t=a_t-r^U_t.
\]

**Proposition 2 (canonical responsibility coordinate).** The map above is
causal, reconstructs (a_t) exactly, and is invariant to every additive gauge
transform of Proposition 1. If the environment consumes only (r^U_t+r^L_t), it
also preserves return pathwise.

*Proof.* Causality follows from (P)'s input restriction to (a_{1:t}). Exact
reconstruction is immediate by definition. Gauge-transformed components have
the same total (a), so both responsibility outputs are unchanged. Their sum is
the original total; Proposition 1 then gives pathwise return invariance.
(\square)

The shared implementation exposes this operator as `CausalGaugeFixer`. A
partial-strength transaction interpolates toward the canonical coordinate while
preserving the total; only full strength is called gauge fixed. This layer was
implemented after the frozen v14.29 study and is not part of that confirmatory
claim.

### 3.5 Violation snapshots

For each frozen path panel, the implementation converts the registered
frequency endpoints and reward floor into three diagnostics: a non-negative
aggregate frequency-violation merit (M), the largest normalized frequency
violation (V), and the number of reward-floor violations (N_R). The exact
endpoint budgets and normalization are fixed by the protocol artifact rather
than estimated from validation outcomes.

## 4. Guarded Restoration Portfolio

### 4.1 Candidate transactions

The frozen v14.29 portfolio contains 15 transactions applied to one trained
anchor: five output-bias actor steps with RMS magnitudes
(10^{-7},10^{-6},10^{-5},3\times10^{-5},10^{-4}), and ten causal router
strengths from 0.0 to 1.0 in increments of 0.1 excluding the anchor strength
0.5. Actor directions are estimated from paired, orthogonal finite-difference
interventions and an action-cost critic. Router candidates change the registered
responsibility assignment but are required to preserve the paired physical and
policy traces exactly.

### 4.2 Design-fold eligibility and selection

Let candidate (c) be compared with anchor (0). On the pooled design panel and
on each of two predeclared folds, it is eligible only if

\[
N_R(c)=0,\qquad M(c)\le(1-10^{-4})M(0),\qquad V(c)\le3V(0).
\]

A router is additionally ineligible unless executed-action, reward, and latent-
policy trace identifiers match the paired anchor on every design and validation
path. Eligible candidates are ranked lexicographically by reward violations,
aggregate merit, worst violation, frozen candidate priority, and registry order.
If no candidate is eligible, the cell abstains.

The selected transaction is then evaluated once on 128 disjoint validation
paths: four disturbance modes crossed with 32 roots. It is supported only if the
same reward, merit, and funnel gate passes. Validation does not trigger candidate
reselection.

### 4.3 Confirmatory decision

One fresh optimizer-seed anchor is the statistical unit. Sixteen anchors are
trained independently in each environment. Abstention and validation failure
both count as failure. An environment passes if the lower endpoint of a
two-sided 95% Wilson interval [@wilson1927interval] for the success probability
is strictly greater than 0.5. The algorithm-level claim requires all three
environments to pass.

## 5. Experimental Protocol

### 5.1 MuJoCo control

We use HalfCheetah-v5, Hopper-v5, and Walker2d-v5 from the MuJoCo continuous-
control suite [@todorov2012mujoco]. Each evaluation spans 1,000 primitive steps
and includes registered standard, low-frequency, high-frequency, mixed, or
out-of-distribution disturbance modes according to the experiment version.

The v12 responsibility experiment compares the full method with its registered
no-leakage comparator using 24 optimizer replicates per arm and environment and
40 held-out paths per cell. The v13 experiment uses fresh optimizer, training,
selection, and evaluation seeds and adds raw lower-frequency and upper high-
frequency gates. Both use 50,000 hierarchical bootstrap draws and family-wise
alpha 0.05; paths are nested within optimizer replicates.

For v14.29, 48 fresh anchors were frozen before portfolio evaluation. Each cell
uses 320 critic-training intervention paths, 320 critic-holdout intervention
paths, 128 design paths, and 128 disjoint validation paths. The source revision,
source manifest, candidate order, seeds, root roles, scheduler contract, and
decision rule were frozen before outcome access. All 48 cells completed.

### 5.2 Synthetic time-series control

Quant v7.4 evaluates 12 learned variants across six registered scenarios. It
uses 24 independent training replicates and eight held-out paths per replicate,
for 13,824 raw rows. Paths and scenarios are averaged within each training
replicate to avoid pseudoreplication. The 12 primary return and lower-LF drift
contrasts compare Freq-HRL with matched flat PPO, flat GRU-PPO, generic HRL-PPO,
generic HRL-GRU-PPO, SAC, and TD3. Holm's procedure controls the primary family
[@holm1979multiple]. This is synthetic control evidence, not live-market or
execution evidence.

### 5.3 Evidence governance

Every manuscript result must be present in the fail-closed authoritative
registry. The registry stores allowed and forbidden wording, stage, decision,
paper-use status, source artifacts, and expected file digests. Development
screens cannot become headline evidence by being rerun at larger scale. The
retired manuscript and legacy C1--C9 matrices are excluded.

## 6. Results

### 6.1 Responsibility-space separation with return noninferiority

The v12 primary family passed in all three environments (Table 1). Freq-HRL
reduced responsibility-space lower-LF drift by 78.2--89.6%, and the family-wise
return noninferiority gate passed in every environment. Positive return
differences are exploratory because superiority was not the registered return
claim.

**Table 1. MuJoCo v12 responsibility-space confirmatory result.** Intervals are
95% bootstrap intervals; all registered family-wise gates passed.

| Environment | Return difference [95% CI] | Responsibility drift reduction [95% CI] |
|---|---:|---:|
| HalfCheetah-v5 | 118.70 [53.01, 198.84] | 89.60% [79.10%, 94.26%] |
| Hopper-v5 | 20.24 [5.19, 38.99] | 78.23% [69.90%, 85.28%] |
| Walker2d-v5 | 31.40 [17.65, 46.18] | 89.56% [85.13%, 93.55%] |

### 6.2 The stronger raw behavioral claim fails

The fresh-seed v13 follow-up passed return noninferiority and responsibility-
space reduction in all environments, but its joint behavioral family failed
(Table 2). HalfCheetah's raw lower-frequency reduction was 5.22% and missed the
registered reduction gate. Hopper reduced raw lower-frequency drift by 58.04%
but exceeded the upper-HF budget. Only Walker2d passed all gates. This negative
result rules out the statement that Freq-HRL universally separates the physical
upper and lower action streams.

**Table 2. MuJoCo v13 stricter behavioral confirmation.** Parentheses indicate
the registered gate decision.

| Environment | Return NI | Responsibility reduction | Raw lower-LF reduction | Upper-HF RMS | Joint gate |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | pass | 74.89% (pass) | 5.22% (fail) | 0.0634 (pass) | fail |
| Hopper-v5 | pass | 73.15% (pass) | 58.04% (pass) | 0.1068 (fail) | fail |
| Walker2d-v5 | pass | 90.37% (pass) | 38.39% (pass) | 0.0240 (pass) | pass |

### 6.3 Fresh-seed guarded restoration is reliable on the frozen panel

The v14.29 portfolio passed its preregistered environment-level decision in all
three tasks (Table 3). HalfCheetah and Hopper each supported all 16 optimizer
replicates. Walker2d supported 15 and abstained once because no candidate passed
the design gate. All lower Wilson bounds exceeded 0.5.

**Table 3. MuJoCo v14.29 guarded restoration confirmation.** Confidence
intervals are two-sided 95% Wilson intervals over optimizer-seed cells.

| Environment | Supported | Success rate [95% CI] | Selected mechanism | Validation merit reduction |
|---|---:|---:|---|---:|
| HalfCheetah-v5 | 16/16 | 1.000 [0.806, 1.000] | 16 routers | 38.04--60.00% |
| Hopper-v5 | 16/16 | 1.000 [0.806, 1.000] | 16 routers | 33.70--60.00% |
| Walker2d-v5 | 15/16 | 0.938 [0.717, 0.989] | 6 routers, 9 actor updates | 0.053--43.54% |

All 38 selected routers exactly matched paired executed-action, reward, and
latent-policy traces on design and validation paths. All 47 supported cells had
zero reward-floor violations. The result is therefore strongest as evidence
that the frozen portfolio can restore the registered responsibility contract
without observed reward-floor harm. For HalfCheetah and Hopper, it does not show
that a learned policy or physical trajectory improved: selection always chose a
function-preserving router. Walker2d required actor changes in nine cells, so the
claim also cannot be reduced to a universal router-only mechanism.

### 6.4 Matched time-series baselines give a mixed performance result

Quant v7.4 supported improvement in eight of 12 Holm-controlled contrasts
(Table 4). It improved both registered endpoints relative to flat PPO, flat
GRU-PPO, and TD3, and improved return relative to generic HRL-PPO. However,
return was significantly worse than generic HRL-GRU-PPO. Return versus SAC and
drift versus both generic HRL variants were inconclusive.

**Table 4. Quant v7.4 pooled matched-baseline contrasts.** Positive directional
deltas favor Freq-HRL after metric orientation.

| Comparator | Endpoint | Directional delta | Holm-adjusted p | Decision |
|---|---|---:|---:|---|
| flat PPO | return | 0.005033 | 0.00042 | improvement |
| flat PPO | lower-LF drift | 0.004775 | 0.00006 | improvement |
| flat GRU-PPO | return | 0.004211 | 0.01506 | improvement |
| flat GRU-PPO | lower-LF drift | 0.003928 | 0.00006 | improvement |
| generic HRL-PPO | return | 0.003449 | 0.00028 | improvement |
| generic HRL-PPO | lower-LF drift | -0.000006 | 0.65460 | inconclusive |
| generic HRL-GRU-PPO | return | -0.002407 | 0.01506 | harm |
| generic HRL-GRU-PPO | lower-LF drift | 0.000033 | 0.65460 | inconclusive |
| SAC | return | 0.000967 | 0.65460 | inconclusive |
| SAC | lower-LF drift | 0.007259 | 0.00006 | improvement |
| TD3 | return | 0.002710 | 0.01674 | improvement |
| TD3 | lower-LF drift | 0.012716 | 0.00006 | improvement |

## 7. Discussion

### 7.1 What the positive result establishes

The v14.29 result establishes a reproducible selection contract: across fresh
trained anchors, a fixed portfolio usually found a transaction that reduced the
registered violation merit on disjoint paths while preserving the reward floor.
Counted abstention prevented forced success, and the seed rather than the path
was the statistical unit. Exact trace matching makes the router result unusually
auditable: its physical behavior is known to be unchanged, so any claim can be
limited precisely to responsibility coordinates.

### 7.2 Why function preservation is both a strength and a limitation

Function-preserving routing isolates a mechanism and removes reward confounding.
It also exposes the central limitation of the current algorithm. In 32 of 32
HalfCheetah and Hopper cells, the selected transaction changed attribution but
not behavior. A reviewer interested in control improvement can reasonably view
this as reparameterization rather than policy learning. The v13 raw-behavior
failure reinforces that concern. Three subsequent single-optimizer-seed
development preflights attempted causal output-head distillation, bounded
distillation, and a multi-source teacher. Only Hopper passed each joint
development gate; HalfCheetah and Walker2d did not support expansion. These
outcomes are excluded from confirmatory claims, but they reject a larger search
over the same post-hoc output-head mechanism. The registered v16--v17.4
development sequence then tested training-time EMA, audit-adaptive, macro-hold,
zero-DC, headroom-homotopy, smooth-macro, audit-optimal macro, and streaming
audit gauges. No
design met its frozen cross-environment expansion rule. Most diagnostically,
v17.2 exactly preserved reward, executed-action, and latent-policy traces on all
360 paired paths, yet no tested smoothing coefficient reduced the registered
lower-LPF32 or joint frequency merit in any environment. V17.3 then optimized
an affine HPF8/LPF32 objective directly and preserved the same three traces on
all 120 paired paths. It reduced lower-LPF32 and joint merit in Hopper and
Walker2d, but increased upper-HPF8 in all three environments and worsened every
frequency endpoint in HalfCheetah. Together these results reject both the naive
EMA target and the reset finite-horizon persistence surrogate as general
leakage solutions. V17.4 then carried the complete HPF8/LPF32 state across macro
boundaries and replanned after every realized total action. It exactly preserved
all 120 paired paths, reduced lower-LPF32 by 60.3--90.4% and normalized joint
merit by 41.2--88.0% in all three environments, and met the absolute upper-HPF8
budget in all three. The strict rule still rejected expansion: only Walker2d
met the absolute lower-LPF32 budget, and HalfCheetah missed the registered
upper-budget feasibility threshold. This identifies the next issue more
precisely. A learned-policy experiment must model the joint physical feasible
envelope and penalize excess above its unavoidable floor rather than assume
that two fixed absolute component budgets are jointly attainable.
The subsequent v17.5 development diagnostic tested that current-step floor on
the rejected v17.4 paths. Although it eliminated local normalized regret in all
three environments, it improved full-trajectory lower-LPF32 in only one and
upper-HPF8 in none; Hopper and Walker2d lower drift increased by 45.0% and
30.9%, respectively. All v17.5 closed-loop traces diverged because responsibility
history enters the policy state. We therefore do not advance v17.5 or treat a
greedy current-step floor as evidence of trajectory-level feasibility. V17.6
then solved the bounded frozen-total-action HPF8/LPF32 allocation over each
complete trajectory. The oracle recovered 81 of the 88 paths that were
infeasible under the v17.4 online split, including every HalfCheetah failure and
all eight Walker2d failures. Seven Hopper paths remained infeasible: their
minimum lower-LPF32 power subject to the upper budget still exceeded the lower
budget. This result separates two development targets. A causal router must
approximate the recoverable full-horizon allocation, while the actor must change
the total-action spectrum on the irreducible Hopper subset. V17.6 reused
rejected paths and is acausal, so it is mechanism diagnosis rather than
performance evidence.

### 7.3 Negative results define the claim boundary

The v13 failure is not a minor ablation. It shows that responsibility-space
success does not imply raw action separation. The Quant harm against generic
HRL-GRU-PPO likewise rules out a uniform performance claim. Reporting these
results makes the contribution narrower, but scientifically identifiable: the
current evidence concerns auditability and guarded responsibility restoration,
not state-of-the-art return.

## 8. Limitations

First, the v14.29 confirmation is conditional on one frozen panel of disturbance
modes and validation roots. The validation paths nested within a seed are not
independent replicates. Second, most successful transactions were
function-preserving routers; they do not demonstrate physical control
improvement. Third, the stricter raw behavioral claim failed in two of three
MuJoCo tasks, and the v15--v17.6 follow-ups remain development-only; several
used only one development optimizer seed. Fourth, Quant is a synthetic
time-series control environment and
contains one supported performance harm. Fifth, Transit, public passenger data,
and order-book adapters exist in the repository but currently lack reportable
records in the authoritative ledger and are therefore excluded from the paper's
empirical claims. Sixth, the reward floor is an empirical finite-panel gate, not
a global safety or convergence guarantee. Finally, the current evidence does
not support deployment, universal frequency separation, or domain-general
performance superiority.

## 9. Reproducibility and Evidence Availability

The v14.29 algorithm is frozen at Git revision
`fc7fa8d8c1e55325af9cb32efece3e0cfc2bbd3c`; its Freq-HRL source manifest is
`02f3ba95376021dff0aa11f30d46dd6159e63b55a1d2678d6011ea350745af39`.
The preregistration, fresh-anchor qualification, cell-level CSV, structured
decision, and report are stored under
`transit_hrl/results/authoritative_evidence_sources_20260830/mujoco_v14_29/`.
The manuscript claim ledger is
`transit_hrl/evidence/authoritative_registry_v1.json`. It verifies source-file
digests and excludes unregistered development outputs. The fresh anchor run used
scheduler tasks `t84875`--`t84922`; the portfolio confirmation used
`t84930`--`t84977` with dynamic placement on node001--node006. Raw checkpoints
and large run directories are intentionally not part of the paper source tree;
the frozen seeds, manifests, summaries, and launch contracts are tracked.

## 10. Conclusion

Freq-HRL makes a normally implicit HRL assumption testable: upper and lower
levels should assume distinct frequency responsibilities. The registered
experiments show that responsibility-space drift can be reduced without
violating a paired reward floor and that a guarded portfolio reproduces this
restoration across fresh MuJoCo optimizer seeds. They also show where the method
does not yet hold: raw physical separation fails in two tasks, and synthetic
time-series performance is mixed. The defensible contribution is therefore an
auditable responsibility contract and guarded restoration protocol. Converting
that contract into a training-time gauge-fixed hierarchy with competitive raw
behavior and return remains the primary algorithmic problem.

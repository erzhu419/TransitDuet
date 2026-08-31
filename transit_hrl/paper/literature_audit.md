# Freq-HRL Literature and Novelty Audit

Audit date: 2026-08-31

Scope: primary conference, journal, and author-hosted records for work directly
adjacent to Freq-HRL's current claim. The audit covers temporal hierarchy,
adaptive decision frequency, action repetition, action smoothness, spectral
representation, and constrained policy updates. It is not a general survey of
hierarchical reinforcement learning.

## Search families

1. Temporal abstraction and multi-level actor-critic: options, Option-Critic,
   FeUdal Networks, HAC, HIRO, and controlled-effect HRL.
2. Learned decision duration and time discretization: FiGAR, time-aware
   Q-learning, TAAC, safe action repetition, and TEMPLE.
3. Physical action smoothness: CAPS, Policy Inertia Controller, LipsNet, and
   LipsNet++.
4. Frequency-aware state representation: Fourier bases and wavelet predictive
   representations.
5. Reward- or constraint-guarded updates: PPO and CPO.

## Novelty boundary

The defensible distinction is not that Freq-HRL is the first method to use two
time scales, filters, Fourier features, or action-frequency penalties. Those
components all have close prior art.

Freq-HRL's narrower object is the *allocation of an action effect between
hierarchy levels*. It distinguishes three questions that prior work often treats
separately:

- **Decision timing:** when an upper policy or repeated command is refreshed.
- **Physical smoothness:** whether the total executed action fluctuates.
- **Responsibility attribution:** which hierarchy level is assigned the slow
  and fast components of the action effect.

The current evidence supports an audit and guarded restoration protocol for the
third object. A function-preserving responsibility router can improve that
coordinate without changing physical behavior, so it must not be described as
an action-smoothing result. Conversely, CAPS, PIC, LipsNet, and LipsNet++ act on
physical policy smoothness and remain relevant comparators for any future raw
separation claim.

## Primary records checked

- FeUdal Networks: <https://proceedings.mlr.press/v70/vezhnevets17a.html>
- HAC: <https://openreview.net/forum?id=ryzECoAcY7>
- FiGAR: <https://openreview.net/forum?id=B1GOWV5eg>
- Time-discretization-aware Q-learning:
  <https://proceedings.mlr.press/v97/tallec19a.html>
- TAAC:
  <https://proceedings.neurips.cc/paper/2021/hash/f337d999d9ad116a7b4f3d409fcc6480-Abstract.html>
- Safe Action Repetition:
  <https://proceedings.neurips.cc/paper/2021/hash/024677efb8e4aee2eaeef17b54695bbe-Abstract.html>
- TEMPLE: <https://arxiv.org/abs/2002.02080>
- CAPS: <https://doi.org/10.1109/ICRA48506.2021.9561138>
- Policy Inertia Controller:
  <https://doi.org/10.1609/aaai.v35i8.16864>
- LipsNet: <https://proceedings.mlr.press/v202/song23b.html>
- LipsNet++: <https://proceedings.mlr.press/v267/song25a.html>

## Experimental consequence

The confirmatory evidence currently compares Freq-HRL with matched learned
hierarchies and flat/off-policy controls, but it does not contain a frozen,
capacity-matched CAPS, PIC, LipsNet, or LipsNet++ baseline on the raw physical
separation endpoints. Related-work citations do not close that gap. A future raw
frequency-separation submission should preregister at least one regularization
baseline and one architecture-level smooth-control baseline, then evaluate
return, executed-action traces, lower low-frequency power, and upper
high-frequency power on fresh paths.

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

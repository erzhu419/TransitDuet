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

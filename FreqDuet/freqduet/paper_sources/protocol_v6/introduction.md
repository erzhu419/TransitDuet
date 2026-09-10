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

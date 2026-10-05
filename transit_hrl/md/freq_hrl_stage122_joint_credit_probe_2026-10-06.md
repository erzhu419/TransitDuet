# Stage122: Frozen Joint Credit Probe

Stage121 joint PPO failed all ten registered gain endpoints. This probe asks
whether its full-episode score is dominated by an inaccurate time-to-go critic,
before replacing the optimizer or expanding the training budget.

Use the first two registered roots410011/410023, both periods50/100, and exactly
the Stage121 first training round:8scenarios x2independent policy-noise paths,
all three methods, horizon1200.96episodes/115,200steps per root; no final weights
loaded, no updates, checkpoints or trace files. Four workers+parent/8GiB,
scheduler dynamic node001-node006. Only scalar JSON leaves the server.

Compare normalized PPO loss gradients for the original MC-minus-critic signal
and MC-minus-phase-baseline. The alternative baseline at each decision index
averages the other seven scenarios within the same noise fold: no own scenario,
no partner noise fold, no evaluation trajectories. A/B noise gradients thus
remain conditionally independent given the fixed exogenous scenario roster.
Also record advantage variance, temporal-mean variance fraction, critic MSE,
gradient norm, cosine and initial sampled-upper return cost. Actor, critic and
teacher must remain exactly unchanged.

This is a diagnosis, not a training intervention or performance confirmation.
Lower variance or positive score cosine alone is not sufficient to adopt a new
policy: a subsequent fresh native direction/update test would still be required.
Stage121 results and all earlier HOLD/gain gates remain unchanged.

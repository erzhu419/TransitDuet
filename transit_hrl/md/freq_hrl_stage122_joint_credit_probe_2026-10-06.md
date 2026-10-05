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

## Run Receipt

Code `8835182e99`; four probe tests and six Stage121 regression tests passed.
Run `pointmaze_joint_reference_credit_stage122_probe_20261006_r1`:
`t135869` root410011/node004 and `t135870` root410023/node006 are DONE.
No aggregator or full performance run submitted. Scheduler commands and fixed
rosters are recorded in the compact preregistration; only result JSON is fetched.

## Result And Next Step

Both roots passed:192episodes/230,400steps, zero updates/checkpoints/traces.
Original joint advantage variance is94.5%-95.9% temporal at upper decisions.
LOO reduces upper advantage variance by92.9%-94.8%, but gradient agreement is
not repaired at both periods:

| Root | Period | Original upper cosine | LOO upper cosine |
| --- | --- | ---: | ---: |
| 410011 | 50 | -0.168557 | 0.231970 |
| 410023 | 50 | 0.226038 | 0.376069 |
| 410011 | 100 | 0.009390 | -0.000335 |
| 410023 | 100 | -0.073273 | -0.071681 |

LOO lower agreement also does not consistently improve. Its zero phase mean
is algebraic centering, not independent evidence of correct control credit.
Initial sampled upper already loses mean0.858976/3.151868 return at periods
50/100 before any optimizer update. Baseline error is real, but simply reducing
advantage variance is insufficient; do not adopt LOO joint training from this.
Stage123 tests paired one-option native full-suffix derivatives on the exact
Stage121 reference channel with the strong lower frozen and fresh query paths.

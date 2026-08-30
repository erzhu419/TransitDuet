# MuJoCo v17.10 Horizon-Reservoir FIR Design

## Purpose

V17.9 met every numerical, physical, upper-budget, Walker preservation, and
mean-lower gate, but recovered 0/33 Hopper failures. The strict prefix budget
dominated all eight FIR candidates. Hopper paths last 82--85 steps, while a
known feasible full-horizon split can borrow upper high-frequency energy over
early prefixes and return it at termination.

## Causal Energy Envelope

V17.10 gives the online router a fixed registered minimum certification horizon
`H_min`. At step `t`, cumulative upper HPF8 energy may not exceed
`max(t+1, H_min) * action_dim * budget^2`. Before `H_min`, this is a finite
energy reservoir. At and after `H_min`, it becomes the ordinary prefix-average
budget. The router remains causal: `H_min` is fixed before the path, and the
projection uses only current and past total/upper actions.

If a trajectory terminates before `H_min`, the endpoint guarantee is invalid
and that path fails closed even if its measured endpoint happens to pass. The
mechanism therefore changes timing, not the final upper budget or its metric.

## Frozen Screen

The candidate set crosses FIR widths 48 and 64, normalized ridge penalties
`1e-5` and `1e-3`, and reserve horizons 0, 16, 32, 48, 64, 72, 80, and 82.
Output gain remains one. The horizon set is bounded by the minimum observed
length in the already reused panel; it is development selection, not a
population guarantee.

All v17.9 gates remain. In addition, every selected-candidate path must meet the
minimum-horizon certification contract. Only a complete grouped out-of-fold
pass allows the previously frozen fresh seed panel to be accessed; fresh paths
shorter than the selected horizon fail the fresh gate.

## Nonclaim

Passing the reused screen would identify a fixed-total-action causal router for
fresh validation. It would not establish reward improvement, actor feasibility,
closed-loop learning, or manuscript support.

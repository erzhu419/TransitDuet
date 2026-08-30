# MuJoCo v17.9 Prefix-HPF FIR Outcome

## Decision

Status: `prefix_hpf_fir_stopped_before_fresh_path_access`.

The selected `prefix_hpf_fir_w64_ridge1e-03_gain1.00` candidate was valid and
met the endpoint upper budget on all 120 grouped out-of-fold paths. It recovered
48/81 oracle-recoverable failures and preserved all 32 baseline-feasible
Walker2d paths. Mean lower power improved over v17.4 in every environment.

The environment split was decisive: HalfCheetah recovered 40/40, Walker2d
recovered 8/8, and Hopper recovered 0/33. All eight candidates had the same
48-path recovery boundary after the strict prefix projection. The total and
environment recovery gates failed, so no frozen fresh-validation path was
accessed.

## Mechanism Finding

The high-frequency-only projection fixed the v17.8 Walker failure without
sacrificing upper compliance. The remaining failure is not ridge width or
regularization. Hopper episodes end after 82--85 steps, and the inspected
full-horizon oracle borrows upper high-frequency energy over most early
prefixes before returning below the endpoint budget. A strict budget at every
prefix forbids that trajectory even though its endpoint is feasible.

The next causal screen may use an explicit minimum certification horizon as an
energy reservoir. Any such contract must fail closed when an episode terminates
before that registered horizon. If this cannot recover Hopper without weakening
the claim boundary, the remaining work belongs in the actor's total-action
constraint rather than another router filter.

## Claim Boundary

This is reused-path grouped development evidence with a frozen total action.
It does not establish fresh-seed generalization, reward improvement, closed-loop
learning, leakage no-tradeoff, or a manuscript performance claim.

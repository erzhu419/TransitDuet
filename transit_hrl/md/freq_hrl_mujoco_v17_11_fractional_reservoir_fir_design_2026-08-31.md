# MuJoCo v17.11 Fractional-Reservoir FIR Design

## Final Router-Only Screen

V17.10 showed that a full 82-step reservoir can recover 63/81 failures but
spends future credit too aggressively: seven paths cannot remain inside the
causal envelope and Hopper reaches only 15/33 recovery. V17.11 is the final
router-only filter screen on this fixed total-action panel.

For minimum horizon `H` and borrow fraction `rho`, the cumulative upper-energy
envelope at step `t` is proportional to

`t + 1 + rho * max(H - t - 1, 0)`.

`rho=1` reproduces the v17.10 full reservoir; smaller values release only part
of remaining credit, while the extra credit still reaches zero by `H`. The
rule is causal and leaves the endpoint budget unchanged.

## Frozen Candidates and Gate

Candidates cross horizons 64, 72, 80, and 82; borrow fractions 0.10, 0.25,
0.50, 0.75, and 1.00; and FIR widths 48 and 64. Ridge is fixed at `1e-3` and
gain at one from the prior grouped screens, giving 40 candidates.

The selected path must remain envelope-feasible and reach its horizon on all
120 paths. The inherited gate still requires 65/81 total recovery, environment
minima 32/40 HalfCheetah, 24/33 Hopper, and 6/8 Walker2d, 30/32 Walker
preservation, upper compliance everywhere, and no worse mean lower power in
every environment.

Passing opens the already frozen fresh panel. Failure ends this router family;
the remaining Hopper gap then requires actor-level total-action feasibility,
not another filter parameter sweep.

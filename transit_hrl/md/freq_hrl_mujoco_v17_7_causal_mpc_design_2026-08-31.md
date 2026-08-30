# MuJoCo v17.7 Causal Responsibility MPC Design

## Purpose

V17.6 showed that 81 of 88 v17.4 joint-budget failures are recoverable by
changing only the responsibility split, while seven Hopper paths require a
different total-action trajectory. V17.7 addresses the first mechanism and
produces an actor-floor signal for the second.

## Causal Contract

At primitive step `t`, the planner may use the realized total actions through
`t`, prior canonical upper/lower responsibility, fixed HPF8/LPF32 definitions,
and registered component limits. It may not use a future observation, future
total action, termination time, reward, or v17.6 oracle action.

The planner:

1. forms a hold or damped-velocity total-action forecast;
2. constructs the exact finite-horizon FIR residual system conditional on the
   realized prefix;
3. minimizes forecast lower-LPF32 power under the forecast upper-HPF8 energy
   budget and component boxes;
4. projects the first upper residual onto the cumulative prefix-energy ball;
5. executes only that first split and replans after the next total action.

The first-step projection guarantees the registered upper budget for every
feasible realized prefix. If the physical component box cannot satisfy that
prefix budget, the planner reports the unavoidable violation instead of hiding
it in a normalized objective.

## Frozen Development Sweep

The reused v17.4 path panel is permitted only for mechanism selection. The
candidate set is:

| Candidate | Forecast | Horizon | Budget ledger | Prefix upper projection |
|---|---|---:|---|---|
| `hold_h16` | hold | 16 | yes | yes |
| `hold_h32` | hold | 32 | yes | yes |
| `velocity_h16` | damped velocity | 16 | yes | yes |
| `velocity_h32` | damped velocity | 32 | yes | yes |

All candidates use velocity alpha `0.25`, decay `0.75`, 24 coordinate sweeps,
8 multiplier bisection steps, power tolerance `1e-8`, HPF8 RMS budget `0.075`,
and LPF32 RMS budget `0.0475`.

Selection is lexicographic: numerical validity, upper-budget path count, joint-
feasible path count, mean lower power, then runtime. No reward comparison is
permitted because total action is frozen.

## Advancement Gate

A selected planner advances to a fresh learned-policy experiment only if all
conditions hold on the 120 reused paths:

- exact reconstruction and component bounds on every step;
- endpoint upper-HPF8 budget on all 120 paths;
- at least 65 of the 81 oracle-recoverable failures become jointly feasible;
- recovery in at least 32/40 HalfCheetah, 24/33 Hopper, and 6/8 Walker2d
  oracle-recoverable paths;
- at least 30/32 already feasible Walker2d paths remain jointly feasible;
- mean lower-LPF32 power is no worse than v17.4 in every environment.

Failure stops v17.7 before fresh training. Passing allows a new frozen
optimizer-seed campaign, but does not itself support a manuscript performance
claim.

## Actor-Floor Boundary

The finite-horizon lower floor under the upper budget is emitted as a causal
cost candidate. It is not yet injected into PPO in this diagnostic. Actor-level
primal-dual training requires a separate frozen experiment after the router
gate passes and must be judged on fresh optimizer and evaluation seeds.

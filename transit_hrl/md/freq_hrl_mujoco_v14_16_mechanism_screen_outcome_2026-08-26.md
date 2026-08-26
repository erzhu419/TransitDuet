# MuJoCo v14.16 Mechanism Screen Outcome

Date: 2026-08-26  
Run: `mujoco_v14_16_crossed_restoration_mechanism_development_20260810_r5`  
Role: development mechanism diagnosis, not confirmation

## Integrity and recovery

- The frozen matrix contains 9 anchors and 72 continuation cells: 81/81 cells merged successfully.
- The two-week server outage left 15 original HalfCheetah attempts terminally failed. Scheduler reroute created one successful replacement for each signature (`t83994` through `t84009`).
- Nine completed anchor records had been retired from live scheduler status and were recovered from the read-only archived queue snapshot. The compact recovery snapshot and its SHA-256 are recorded in the run-scoped sync manifest.
- The sync layer accepted a reroute only when a signature had exactly one successful producer. It would fail closed if two distinct attempts both completed.

## Registered outcome

The preregistered primary arm, `l2_path_freeze_crossreplay`, is not ready:

- engineering gate: 0/9 cells;
- complete effect gate: 0/9 cells;
- complete environments: 0/3;
- trained checkpoint selected: 1/9; fallback checkpoint retained: 8/9;
- pooled normalized return effect: +0.001698;
- pooled log reductions: LowerLF -0.000982 and UpperHF -0.026738, both in the wrong direction.

The best diagnostic arm was `l2_path_trainreplay`, without reward-actor freezing or crossed replay:

- engineering gate: 2/9 cells;
- complete effect gate: 2/9 cells;
- complete environments: 1/3 (Hopper only);
- pooled normalized return effect: +0.004955;
- pooled log reductions: LowerLF +0.086516 and UpperHF +0.027919.

The return-preserving `worst_mode_trainreplay` arm had the largest pooled return effect (+0.073637) and 4/9 engineering passes, but no environment passed every endpoint gate.

## Causal diagnosis

1. Binary reward-actor freezing is counterproductive. It removes the policy improvement direction while the frequency projection alone does not reliably change closed-loop state occupancy.
2. Hard all-path feasibility is too brittle for this optimizer. Five frequency endpoints across 16 guard paths create 80 simultaneous frequency constraints; most pathwise cells never found a trained checkpoint with zero violations.
3. Crossed frozen-state replay did not repair the mismatch between fixed-state action projection and closed-loop rollout endpoints.
4. The non-frozen pathwise arm shows that the L2 direction can work in Hopper, but it does not generalize to HalfCheetah or Walker2d. More seeds would quantify an already rejected mechanism rather than repair it.

## Successor requirement

The next protocol must replace binary freezing and hard all-path feasibility with an occupancy-aware constrained update:

- train upper and lower leakage cost critics from native rolling frequency costs;
- use primal-dual policy updates so reward and leakage gradients coexist;
- aggregate path risk with a preregistered CVaR objective instead of requiring every noisy path to improve by 5%;
- retain the actual closed-loop reward floor and independent guard seeds;
- compare against both the matched control and the v14.16 non-frozen L2 path arm on fresh optimizer seeds.

No v14.16 result supports a paper claim of cross-environment learned separation, no-tradeoff performance, or confirmation readiness.

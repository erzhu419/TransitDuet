# Stage281: Cross-Future Fleet/Service Information

- Run: `native_transit_component_probe_stage281_20261017_r1`.
- Stage280 found selected-pair fleet optimism plus held service sign error. Replace one component at a time to determine where better information can improve decisions.
- Freeze raw / Linear9 / observed-df risk, raw service, wait feasibility, ties, queries and both banks. Replay all seven Stage279 choices; report the three relevant controls.
- Five probes: MC service crossed with each of the three frozen risks; MC fleet with raw service; both MC components. MC information uses assigned16 or its fixed first8/last8, while frozen-model training is unchanged. Score on opposite16 and its two halves; report component MSE and selection-half action disagreement. Assigned-bank results are in-sample descriptions only.
- Finite-MC means come from action outcomes at the query state, including held states. These are label-informed information probes, not trained policies, deployable estimates or oracle upper bounds. Opposite-bank outcomes never enter proposals; no winner is selected.
- Same 32 cells, 2,048 queries, 84,480 repeated model/state exposures. All train/retained/excluded/full-held/prefix-head-held populations, views and banks remain reported.
- One scheduler CPU / 3,072 MiB, any node001-node006; zero new fits, raw-reference training, simulations or actor updates. Full ledgers stay remote; scalar report only.
- Decision after results: compare component-MSE and physical cost/wait together. Fleet-only benefit points to selected-action risk estimation; service-only benefit points to service ranking; joint-only benefit requires coordinated estimation. No opposite-bank gain means more accurate in-bank sample means alone have not solved the problem.
- Verification: five focused tests cover component units/anchors, feasibility/ties, future isolation, sample-half stability, error denominators, complete cell/group control replay and the completed-parent gate. Both cold CLI startups passed.
- Completed: `t166148`, node006, done/0 in 60.01s, no retries. All 32 cells qualified, raw/cell/group replay errors zero. Pulled 1,538,553 scalar bytes (196,813 transport bytes); no trajectories, models or query ledgers.

## Results And Next
Assigned16 proposals, opposite16 evaluation, mixed pooled, versus frozen raw choices. Cost x1e6; wait in minutes; negative is better.

| Information replacement | Excluded cost | Excluded wait | Held cost | Held wait |
| --- | ---: | ---: | ---: | ---: |
| Fleet only, raw service | +664.946 | -0.004985677 | +109.791 | -0.003338314 |
| Service only, raw risk | -288.789 | -0.009911719 | -643.659 | -0.009253255 |
| Service only, Linear9 risk | -349.238 | -0.010280534 | -745.432 | -0.009709375 |
| Service only, observed-df risk | -327.721 | -0.009628613 | -748.261 | -0.008808919 |
| Both components | +12.487 | -0.008213053 | -588.332 | -0.007661198 |

All three service-only replacements improve pooled cost/wait in both evaluation halves and both bank directions. Raw-risk and observed-risk service replacements also improve cost in all four mixed-view/bank cells at each primary scope; Linear9 loses cost in mixed-a held bank1. Root397-only held costs remain positive for all three, so this is not uniform generalization.
Service replacement does not save fleet: with raw risk, service -975.096/-961.042e-6 outweighs fleet +686.306/+317.383e-6. Service MSE x1e6 falls from 71.492 to 35.050 excluded and 111.846 to 41.128 held. In contrast, direct fleet MC MSE is 893.163/957.943 versus raw 655.806/895.036; the in-bank fleet-only gain -4517.833/-4183.493e-6 becomes opposite-bank harm. Full MC is not an oracle.
Service-only gains also survive either fixed8 proposal half in both primary pooled scopes. Joint-MC proposal halves disagree on 35.8%/30.5% of actions versus 11.5%/8.8% for service-only observed risk. Do not select a half or risk arm from these outcomes.
Next Stage282: learn service residuals only, preserving the frozen risk heads and feasibility. Reuse Linear9 and matched-df action/observed maps and penalties; training-only service RMS clip, with uncut controls. Cross each service predictor with all three risks; retain the frozen MC-service benchmark. No raw/risk refit or new simulations.
Limitations: reused development single-intervention outcomes and repeated exposures, not full HRL or seed CI. MC query labels provide information diagnostics, not deployed estimates. Stage253/248/169 remain closed; defaults unchanged.

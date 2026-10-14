# Stage236: fixed-candidate stability admission contrasts

## Protocol

- Run: `native_transit_stability_admission_stage236_development_20261015_r1`.
- Reuse Stage234 actions and Stage235 curves: 27 controls plus 12 admission variants; zero training/rollouts. Check fixed candidates on both future halves, every prefix deletion, or both. Keep pooled strict improvement; allow subset cost/wait nonharm within `1e-12`, with nonincreased risk for safe selection. Rejected candidates use raw.

## Results

- t153189 completed on node006; seven tests passed. Legacy choices/scores and 5130 aggregate metrics reproduced exactly. Only the 1.86 MB report was pulled.
- Mixed-source raw-mean held-other cost/wait deltas: future-only **+1.779e-6 / +1.797e-5 min**; prefix-only/joint **-7.370e-7 / -1.563e-6 min**, from only **one** changed action in 960 cases. Joint raw-safe still increased cost **+8.388e-5**. Admission mainly removes interventions, not the tradeoff.
- Jointly admitted `mixed_b/1/even_odd/1`, raw-safe rank3/strength1, worsened cost/wait on outer training states too. Single-source root401 raw-mean joint remained positive; root397 still increased wait. All scopes/source/future contrasts are retained in the report. Defaults unchanged.

## Next

Stop tightening admission. Freeze raw reference/feature coordinates and compare residual refits at 54/60 training states and 2/4 future draws. Track regression error and decision cost/wait on training/held scopes against the changing-reference path. Keep every source cell and the fixed grid.

## Limitations

Correlated development reuse gives no fresh CI. Deletions remove validation records only. Reference-refit and prefix-coverage effects remain confounded; raw fallback is abstention.

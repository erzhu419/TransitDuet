# Stage169: Restore the Fixed-Window Forecast Primitive

Stage168 qualifies 200 episodes but no horizon combination beats window forecast
on both roots. Prefix-wait/full-fleet improves full/full cost by -0.010981/-0.040169
yet loses to window by +0.037369/+0.048339. At 40 initial direction decisions per
root, the five rolling candidates never contain the window baseline's first-six
planned departures. Test candidate coverage before further learning.

Append a fixed-window forecast primitive: change only internal points of the
current six-trip window, retaining its first/last endpoints, next anchor and
tentative tail. Preserve the hybrid prefix-wait/full-fleet score and all five
old candidates. Existing candidates retain precedence on exact score ties.
Add a forced-incumbent condition that must reproduce original window-forecast
physics and schedules throughout each full native episode.

Four conditions: window source, five-candidate source, forced incumbent and
six-candidate augmented pool. Two roots, five regimes, all four registered scenes:
160 episodes, 9,820,800 ticks, zero updates, 80 exact source controls,
40 forced-incumbent reproductions and 120 selected-window ledger qualifications.
Primary is actual augmented-pool cost versus window and five-candidate source;
also report wait/reward, fleet/nonfleet components, choices and proxy margins.

Ten unpinned scheduler tasks on node001-006, 1 CPU/3 GB each. Stage code and fresh
model inputs only. Raw records/checkpoints remain remote. 61 focused tests pass:
262-trip dispatcher fixture/goal reproduction, unchanged original five
scores, fixed-window geometry, candidate selection, control/ledger accounting
and scheduler staging. Run
`native_transit_incumbent_window_stage169_frozen_20261011_r1`, revision
`eac46428b2`: t142440-t142449 are RUNNING on node001/004/005/006. All 80 source
and 40 forced-incumbent native reproductions and performance results are pending.

## Limitations

After earlier rolling choices move anchors, the window primitive uses the current
feasible anchors, not an impossible reset to the historical baseline trajectory.
Adding the primitive cannot guarantee native cost improvement from proxy ranking.
Reused scenes and frozen lower remain mechanism development, not learned HRL or
independent confirmation. Negative outcomes stay in the reported matrix.

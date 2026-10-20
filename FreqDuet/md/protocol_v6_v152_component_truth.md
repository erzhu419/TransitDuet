# V152: read-only true-component accounting

Freeze V151 execution1b783ea8f3 and its sealed prior. Reuse only its384 saved
confirmation paths; no replay, fitting, training or control changes. V146(3s,0s),
3600s window and .1/.1 tails remain fixed. Run through scheduler on node001--006;
return only a small summary, leaving all traces and per-cell data on server.

Reconstruct V149 and V151 component totals against their saved points, then
partition the same physical follower's true query-to-ready clock into service,
running, actual holding and execution wait. Preserve joint ready residence when
its forecast cannot separate holding from execution; opaque reference forecasts
remain outside paired component comparisons. Reproduce original full endpoints.

Compare true component bias/MAE and exact cancellation:
full MAE = sum(component MAE) - 2*cancellation, query by query before averaging.
Use existing episode weighting and crossed actor/day statistics. Primary readout:
all-day, clearance, hour2 and hour13; inspect every clearance launch individually,
especially the four handoff regressions. This is consumed-path mechanism
diagnosis, not another confirmation gate or parameter-selection dataset.

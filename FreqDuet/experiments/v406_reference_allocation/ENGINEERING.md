# V406 Unit Repair

The first 20 paths passed physical and paired-randomness validation. Readout
then exposed a real implementation error: `station.od` is hourly demand, but
`historical_rates` divided it by 60. Event-graph rates are per second, so the
censored-state prior was 60 times its intended value. Those results are kept
as run, not used to assess the intended predictor.

The repair changes only `/60` to `/3600`. A synthetic check now verifies a
360-passenger hourly profile yields 0.1 passenger/second, including interpolation.
No change to forecast windows, scenario multiplier, costs, actions or gates.

Server `FreqDuet-v406-reference-allocation/source01` and `run01` stay untouched.
Corrected source and output are isolated in `source02` and `run02`. Reuse the
eight unaffected `simple_cap` and legacy `cap_10` paths; replay only the twelve
candidate paths on the same four development days. The supplemental cap is
12 starts, no automatic retry. This is a unit-correction comparison, not fresh
confirmation. Its frozen manifest is `unit_repair_protocol.json`.

`review/readout/results.json` contains the original incorrect-unit readout;
`review/unit_readout/results.json` is the completed corrected comparison.
Raw trajectories stay on the server. Current local source includes the repair;
the server source01 snapshot records the original implementation.

The local CLI also removes an unused copied aggregation route that checked
only the legacy cap comparator. `readout.py` is the single result entry and
checks both frozen comparators. This does not change a traffic decision.

# FreqDuet Paper Effect Status

Last updated: 2026-06-26 CST

## Bottom Line

The current paper effect exists, but the claim must stay conservative. The
defensible result is:

FreqDuet is statistically tied with a strong fixed-headway baseline, strongly
beats weaker rule-holding and rule-MPC baselines, and has a clear leakage-control
mechanism signal. It should not claim universal superiority over fixed-headway
or every internal ablation.

## Primary Evidence

Artifacts are in
`FreqDuet/freqduet/results_freqduet/paper_package/current/tables`.

### External Classical Baselines

From `paper_external_classical_v1_ep200_60seed_paired_deltas.csv`, lower
composite is better and deltas are `main - baseline`.

| Comparison | Scope | Composite delta | 95% CI | Win rate | Interpretation |
| --- | --- | ---: | --- | ---: | --- |
| fixed-headway | overall_shared, 240 paired runs | +0.0084 | [-0.0100, +0.0321] | 0.583 | statistically tied with the strong fixed policy |
| rule-holding | overall_shared, 240 paired runs | -0.6010 | [-0.6243, -0.5742] | 1.000 | decisive improvement |
| rule-MPC | overall_shared, 240 paired runs | -1.9865 | [-2.0480, -1.9254] | 1.000 | decisive improvement |

Wait-time against fixed-headway is not worse: wait delta is -0.0717 with 95% CI
[-0.1820, +0.0585]. The fixed-headway tie is driven by CV and overshoot
tradeoffs, not a wait-time failure.

### Internal Mechanism/Ablation

From `paper_ablation_v1_ep200_60seed_paired_deltas.csv`, `noleakage` is the
cleanest mechanism failure. In the terminal domain, main vs noleakage has
composite delta -0.3326, 95% CI [-0.3995, -0.2664], win rate 0.983. The same
direction holds across highnoise, odshift, and rushshift.

The other internal variants (`nofreq`, `rawhistory`, `allfreq`, `nopromotion`)
are close controls in the final paper protocol. They are useful as mechanism
context, but not as a claim that every module dominates every alternative.

## Data/Realism Evidence

The paper package now includes:

- public AFC/APC demand-profile evidence;
- public MTA subway OD-estimate samples;
- public MBTA bus board/alight/onboard-load calibration targets;
- MBTA same-network APC-to-static-GTFS route/stop matching with Route 111 as a
  route-level load calibration target;
- local MBTA live GTFS-RT VehiclePositions/occupancy snapshots;
- derived full-day MBTA SUMO APC/AVL replay snapshots from the H2Oplus/CFCMT
  benchmark.
- FreqDuet-only MTA Bus Time API offline cache with 378 routes, 13,585 stops,
  22,730 route-stop sequence rows, and 144 route-filtered SIRI
  VehicleMonitoring rows.

The same-network audit now supports structural field-calibration readiness plus
AVL realism evidence. It still is not a full field validation: exact same-day
AFC/APC/AVL/OD calibration needs historical AVL and route-level OD for the same
service days as the APC targets. The MTA Bus Time cache is route/stop/AVL
geometry evidence only; it is not APC/onboard-load data and not FreqHRL result
data.

## Safe Claim

FreqDuet provides a frequency-separated HRL policy with leakage control that is
robust, mechanistically traceable, competitive with a strong fixed-headway
baseline, and decisively better than weaker rule/MPC baselines in the current
simulation evidence package.

## Unsafe Claim

Do not claim that FreqDuet robustly beats fixed-headway everywhere, has already
been field-calibrated on exact same-network AFC/APC/AVL data, or has observed
field wait-time improvements.

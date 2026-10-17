# Stage289: waiting-constrained cross-future joint gains

Use only Stage288's 60 training prefixes, 360 states, 24 futures, six actions
and 16 fixed raw-reference profiles. No held rows/labels, fitting, simulation
or actor updates. Raw feasibility and native original fallback are unchanged.

- Primary: futures 0-15 select actions; freeze before opening 16-23. Compare
  finite-MC minimum cost with/without mean wait <= own raw, versus saved raw
  and unchanged Stage247. Zero margin; raw/original/index tie order. Two fixed
  reverse checks hold out 0-7 or 8-15; no direction/winner selection.
- Average profiles and six probes before 60-prefix stratified CI95 (six per
  root/regime, bootstrap5000, seed289); report conditional shared-future MC SE
  separately. Joint retention requires primary cost/wait upper bounds <0 and
  negative cost/wait in both four-draw evaluation halves.

Run: `native_transit_wait_constraint_stage289_20261017_r1`.

## Result and next decision

Scheduler `t173090` completed on node002 with exit0. One CPU/3 GiB, no node
pin. Six focused tests and both CLI startups pass, including independent-bank
harm failing the gate and no uncertainty reduction from duplicate profiles.
Only the 46,204-byte scalar report was pulled; action ledgers remain server-side.
The preregistered joint-retention gate fails. Primary constrained-minus-raw:

| Metric | Selection16 | Independent8 [95% prefix CI] |
| --- | ---: | ---: |
| Cost x1e6 | -3710.48 | 116.76 [-755.20, 1269.90] |
| Fleet x1e6 | -2595.12 | 1038.05 [186.23, 2162.95] |
| Service x1e6 | -1115.35 | -921.29 [-1280.50, -587.37] |
| Wait / min | -0.0105798 | -0.0090018 [-0.0124326, -0.0058859] |

Service and wait retain 82.6% and 85.1% of their selection-point gains. Fleet
flips from predicted savings to an independently evaluated increase. Evaluation
halves have cost -115.98 and +349.49 x1e6, with wait improving in both.
The two fixed reverse checks also improve wait, but cost is -860.59
[-1810.49, 113.84] and +79.27 [-891.82, 1076.80] x1e6: neither supports
joint gains. Constraining wait additionally improves primary wait versus
cost-only (-0.0015361 [-0.0024413, -0.0007290] min), without confirmed cost benefit.

Retain the service/wait information signal, not the selector. Next: separate
proposal, independent fleet-risk screening and independent evaluation, rather
than another wait threshold or larger prefix budget. No further task dispatched.

Limitations: reused development data; the three directions are not independent
replications. This finite-MC information diagnostic is not a deployable policy
or guaranteed oracle bound. Stage288's restricted, single-intervention metrics
are unchanged; new-prefix confirmation remains separate.

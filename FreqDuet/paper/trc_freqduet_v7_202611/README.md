# FreqDuet TR-C v7

[Manuscript PDF](main.pdf) | [Source map](evidence_manifest.json)

This draft reports the V500--V502 branch. It evaluates joint dispatch and
stop holding under a fixed fleet and trip roster. The separate V406/V410
manuscript uses a fixed timetable and answers a matched-holding question.

## Rebuild

From this directory, run `bash build.sh` with Python 3, NumPy, Matplotlib,
ripgrep and TeX Live (including `latexmk` and `elsarticle`). The result is
`build/main.pdf`. The build regenerates the tables from the archived results
and the figures from those results and the online harmonic estimator. It
does not need simulator checkpoints, start traffic paths, or retrain policies.

- [V500](../../experiments/v500_pathb_headroom_gate/PROTOCOL.md): rule-based
  development probe and corridor construction.
- [V501](../../experiments/v501_pathb_b1_stability/PROTOCOL.md): frozen
  checkpoint comparisons.
- [V502](../../experiments/v502_pathb_trc_matrix/PROTOCOL.md): nominal and
  shifted-policy comparisons. The table script preserves means, checks
  archived pairing, and recalculates V502 intervals over matched training
  seeds and days. Original protocol files and results are unchanged.

The archived controller sees exact simulated post-boarding queue counts.
It also accumulates 20-second APC feed contributions using a 60-second
conversion before a one-minute update. A constant held rate of 2 reaches
the filter as 6. The paper reports the results of that implementation.
The unit-consistent curve in Fig. 2 is an estimator illustration, not a
policy reevaluation. The study does not isolate a pure upper/lower routing
swap or establish deployment with vehicle-only observations.

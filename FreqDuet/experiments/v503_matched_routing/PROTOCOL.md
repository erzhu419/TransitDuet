# V503: Unit-consistent APC and matched routing

This tests whether aligned **current harmonic summaries** beat inverted
routing under the same simulation inputs and action constraints. It is not
a reproduction of V502, nor a comparison with unfiltered APC.

## Locked comparison

- In the study copy, scale the held rate by feed tick duration, 20 s,
  not the 60 s filter bin. Remove exact post-boarding queue from the lower
  policy and its simulator context. Do not train either arm from an old model.
- Both arms start identically for a seed: same MBTA-111 world, holding bins,
  lower ledger and 0.50 retention, harmonic estimator and promotion, reward,
  and fifteen upper timetable actions. The ledger uses the same LF signals
  in both arms. This fixes the budget rule, not realized holding under
  differing trajectories.
- For each band, use the same directional sums, spatial RMS, spatial
  standard deviation and aggregate one-bin change at dispatch. At stops use
  the current value, two-stop downstream mean, and their one-bin changes.
  Use the same scale and clipping for both bands. Aligned selects LF for
  the upper policy and signed HF for the lower; swapped reverses them.
  At fixed physical state each policy's other inputs are identical.

## Budget and decision

First one paired smoke: 3 training days and 1 evaluation day per arm.
Check filter units, feature routing, mask, actual optimizer update counts,
common scenario, and all 262 trips. Do not filter by smoke outcome.

Then one MBTA-111 nominal matrix: four paired seeds, 200 training days
each, and 16 untouched common evaluation days per policy at episode 200.
No other checkpoints are evaluated, and no held-out result enters training.
At most 8 smoke paths and 1,728 formal paths, via two smoke and eight formal
scheduler tasks on Linux CPU nodes, high priority. No automatic retries.
Only summary JSON returns locally; checkpoints stay on the server.

Report aligned minus swapped for restricted passenger journey, raw arrival
CV2, readiness delay, holding dose, and all seeds. Use percentile intervals
from paired seed-by-day resampling. A joint win requires a negative upper
endpoint for journey and non-positive ones for CV2 and readiness, with the
full trip roster complete. Coefficients and endpoints are not retuned after
results. A failed joint criterion remains a failure. With four seeds and
one corridor, a pass would be evidence for these specific representations,
not general superiority or proof of a unique frequency mechanism.

V500--V502 remain unchanged. The staged runner is the frozen V502 runtime
with the explicit study patch; generated source is not hand-edited on nodes.

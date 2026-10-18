# V107: frozen quarter-gain rule confirmation

V106 found large passenger/readiness savings at gain 0.25, but did not establish
CV2 nonharm. Keep both forward025 and bilateral025 as co-primary rules; do not
pick a family winner, alter gains, interpolate, fit a CV margin or update actors.
Higher-gain results remain reported V106 development tradeoffs, not new candidates.

Reuse the V106 implementations unchanged. Native B200, V96 preserve and zero
holding are controls; upper weights/deployment background, fleet, clocks,
causal evidence and seven feasible holding bins remain the same. Zero is not
an exact fixed-upper external baseline. Both actors and duals remain frozen.

Use the other ten archived actors (`VALIDATION_SEEDS`, excluding 12037/933083),
and 16 new days 2071001--2071016. They are unseen in V106, not newly trained or
untouched throughout the project's history. Native/preserve retain three paired
samplers; zero and both deterministic rules retain one shared path per actor/day.
Total: 1440 full-day rollouts. No adaptive extension after reading results.

Average samplers before the existing 5000-draw crossed actor/day bootstrap.
Report absolute costs/intervals, paired changes against all three controls,
the direct same-gain family contrast, actor/day axes and evidence coverage.
For each co-primary rule require passenger upper < 0 and CV2, readiness,
unlaunched/incomplete upper <= 0 against BOTH native and preserve. No margins.
The existing two-sided 95% intervals provide 97.5% one-sided upper bounds;
each rule's claim is an intersection of all constraints, with two co-primary
claims. Zero comparisons describe the faster/less-regular tradeoff, not a
required Pareto victory. A crossing-zero interval is inconclusive, not equality.

Smoke: new tests and five exact cached V106 control replays; no new confirmation
day labels. Formal shards wait for this real marker; aggregate waits for every
actual shard. Scheduler on node001--node006 only, dedicated CPU environment;
only markers/final small JSON return locally. A pass does not switch main or
establish a frequency-separation contribution. A failure keeps the measured
tradeoff and ends this fixed confirmation budget, without another gain search.

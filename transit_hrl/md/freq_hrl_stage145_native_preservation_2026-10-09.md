# Stage145: Mechanism-Preserving Native Transit

PointMaze Stage144 remains a positive upper-conditioning diagnostic, not evidence
for frequency routing or joint HRL. That recipe is frozen. The mainline returns
to native Transit without changing its physical controller or mature backend.

## Change

The older `freq_transitduet` runner is not the current FreqDuet baseline. The
generic `TransitFrequencyTracker` also omits historical harmonic priors and OD
features, uses signed linear RLS instead of log-count RLS, and treats RawHistory
as an EMA. It therefore cannot serve as a baseline-preserving generalization.

`native_freqduet` is a separate minimal source copy of current FreqDuet. Its
runner, simulator, RE-SAC policies, actions and rewards are unchanged. Only two
imports redirect the native harmonic estimator/prior fit to the extracted
domain-independent `freq_hrl.encoders.count_harmonic`. The original estimator
remains the test oracle; old experiments are not rewritten.

## Qualification

Focused tests compare historical prior fitting, sparse station/OD bins, native
features, forecast phase, reset, promotion absorption and causal prefixes.
RawHistory must retain actual bins. The qualification comparator must reject a
changed action/reward or a run with no learned updates.

Scheduler preflight uses one seed, two shortened training episodes, one held-out
evaluation episode and explicit smaller replay batches. Full qualification uses
two seeds, the unchanged 30-episode warmup, 32 training episodes and two held-out
evaluations per implementation. Both upper/lower update calls and actor changes
are required. All non-wall episode fields, every action and final network arrays
must match exactly. Temporary arrays stay on the server and are removed after
comparison; only compact JSON is pulled. No checkpoint is produced.

## Next

After native extraction qualifies, register a matched-budget native
RawHistory/correct-routing/Swapped comparison. Preserve physical actions,
reward, clocks and learning backend; isolate frequency responsibility before
returning to promotion, leakage or a second domain. Stage145 itself is an
equivalence gate, not a performance or domain-general claim.

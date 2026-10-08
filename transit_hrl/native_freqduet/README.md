# Preserved Native Transit

Isolated source snapshot of `FreqDuet/freqduet`, copied on 2026-10-09.
The latest reference change touching the runner/estimator was `03c47a401c`.
Only the runner's required modules, base/harmonic configuration and four small
route-input spreadsheets are copied. Results, caches and checkpoints are absent.

The native runner, simulator, physical actions, rewards, clocks and upper/lower
RE-SAC implementations are unchanged. Two frequency imports now use
`freq_hrl.encoders.count_harmonic`; the original `frequency/intensity_estimator.py`
is retained as the extraction test oracle, not the production estimator.
Historical global/local/OD priors, log-count RLS, residual filtering, forecast
phase and native feature layout are preserved. This is separate from the older
`freq_transitduet` experiments and does not replace their archived protocols.

From `transit_hrl`, run the native entry point with `PYTHONPATH=.`. Stage145
first checks estimator and learned-control equivalence. It does not establish
frequency attribution or generalization beyond Transit.

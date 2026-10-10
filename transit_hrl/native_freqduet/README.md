# Preserved Native Transit

Isolated source snapshot of `FreqDuet/freqduet`, copied on 2026-10-09.
The latest reference change touching the runner/estimator was `03c47a401c`.
Only the runner's required modules, base/harmonic configuration and four small
route-input spreadsheets are copied. Results, caches and checkpoints are absent.

Stage145 preserved the native runner, simulator, physical actions, rewards,
clocks and upper/lower RE-SAC implementations. Two frequency imports now use
`freq_hrl.encoders.count_harmonic`; the original `frequency/intensity_estimator.py`
is retained as the extraction test oracle, not the production estimator.
Historical global/local/OD priors, log-count RLS, residual filtering, forecast
phase and native feature layout are preserved. This is separate from the older
`freq_transitduet` experiments and does not replace their archived protocols.

From `transit_hrl`, run the native entry point with `PYTHONPATH=.`. Stage145
first checks estimator and learned-control equivalence. It does not establish
frequency attribution or generalization beyond Transit.

Stage150 changes signed dispatch timing in this copy: channels/haar commit at
nominal launch minus the maximum advance, observing the current clock rather
than future demand. HIRO, warmup, fixed experts, fleet limits and RE-SAC remain
unchanged. Previously a negative command was queried only at nominal launch
and could not advance a departure. The original FreqDuet source is untouched.

Stage152 adds unit action coordinates inside the upper critic only. Physical
actor outputs, replay actions, actuator bounds and rewards are unchanged;
`upper.critic_action_units: seconds` retains the preserved critic. This is a
matched representation experiment, not an adopted performance improvement.

Stage153 tests `upper.weight_reg_mode: physical_sum`: the unit critic's first
action weights and affine bias are expressed in physical coordinates before
L1 is evaluated. It preserves the seconds-coordinate prior for equivalent
functions; ordinary `sum` and `mean` remain separate controls.

Stage155 tests an eight-decision soft Retrace upper backup, storing collection
densities and terminating sequences at episode boundaries. `backup_horizon: 1`
retains the preserved one-step update bit-for-bit. Only the global unmodified
policy trajectory is supported by the native trace path; fixed7 is a frozen
evaluation intervention. This does not adopt a performance improvement.

# Stage189: Selected-Action Optimism

Cached diagnostics t144373 completed exit0, zero fits/updates/episodes.
Use predicted cost/wait, oracle wait only, oracle cost only, and both oracle
values to isolate failures; future-outcome arms are diagnostic only.

Sequence regime/root397 still increases cost with an oracle wait mask:
+0.000666 versus+0.006675 under the learned mask. Wait calibration alone
does not fix cost ranking. Its all-candidate cost bias is-0.000680, but
selected-action bias is-0.018017. Scene/root401 wait bias changes from
+0.002515 across all candidates to-0.019928 after selection.

This supports investigating selection-aware pessimistic cost/wait estimates
and conditional state/action representation before collecting more seeds.
Calibration must be learned inside training groups, not from outer residuals.
Three tests pass. Reports stay compact; the35MB bank/models remain remote.

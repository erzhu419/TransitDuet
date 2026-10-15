# V51: observable turnover-conditioned holding response

After V50's completed-prefix travel screen, extend the same seven-bin softmax
with seven observed features: previous completed-trip availability, mean/last
eligible holding and duration (memory decay 900 seconds), terminal layover,
actual launch delay, and elapsed trip time. Preserve the 4096-row prefix, ridge,
half-pseudocounts and seven executable actions. Train only on action times strictly
before each issued origin; previous-trip features require completed returns.

Keep the real controller frozen. In forecast draws, update holding summaries and
completed-turnover context using predicted actions/returns, not future telemetry.
Smoke tests check future-trip independence, simultaneous-label exclusion and
physical reuse; reproduce V48 passenger_joint metric draws exactly after adding
metadata to the forecast-only event/action interface. Prequential screen requires
MAE and NLL no worse than the base model in each reused-demand cohort.

Then compare the four forecast cells: V48, V50 conditioned_shared travel only,
turnover_legacy holding only, turnover_shared joint. Scheduler-only server runs;
no main-policy promotion without independent action-response validation.

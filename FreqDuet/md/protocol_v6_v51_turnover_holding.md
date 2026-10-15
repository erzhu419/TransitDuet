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

## V51B Revision
The 19-feature refit failed the prequential screen: MAE delta -0.0096 s but NLL
delta +0.0095 [0.0037,0.0207]. No four-cell matrix was launched for that model.
V51B preserves the original seven-bin probabilities and learns only seven
regularized coefficients for an ordered-action exponential tilt. Fit these on
matured predictions issued before each historical decision, not in-sample
probabilities refit at the current origin. With no matured predictions the
correction is zero. Keep the same turnover features, prefix gate and four-cell
comparison; retain V51's negative results in its separate experiment directory.

## V51C Residual Context
V51B also failed: MAE delta +0.0126 s and NLL +0.000305 [0.000131,0.000466].
Keep that negative result. V51C changes the two action-memory coordinates to
previous completed-trip mean/last prequential innovations (actual minus the
strictly earlier issued base prediction). The physical turnover features and
seven-coefficient tilt stay fixed. Forecast returns carry sampled action minus
the base conditional mean into the next trip; no future observed residual is
read. Missing early-day issued predictions have no scored residual memory.

## V51D Boundary Response
V51C's overall gate failed. Phase audit t160148 separates 2,872 first decisions
from 54,162 later decisions: first-decision MAE/NLL deltas -0.130/-0.0106,
later deltas +0.0233/+0.00101. V51D exposes the same residual/physical context
only before the first eligible holding after a completed, scored previous trip.
Fit only those active prefix rows, retaining the existing ridge and seven
coefficients. Later decisions use the base probabilities; all eligible predicted
actions still update the next completed-trip residual history. This is a
post-audit development revision; the original gate and fresh-validation need stay.

V51D smoke t160149 passes 85 tests and the exact V48 draw replay. Preparation
t160150-t160154 and gate t160155 pass all three reused-demand cohorts: overall
single-decision MAE delta -0.0101 s [-0.0135,-0.0065], NLL delta -0.000825
[-0.001096,-0.000555] (six demand clusters).

Four-cell forecasts t160157-t160161 complete all 612 origins; aggregation
t160162 retains the mixed result. Holding-only 30-minute cumulative holding MAE
delta versus V48 is -1.27 s [-1.86,-0.72], but backlog MAE rises 0.137
trip-seconds [0.021,0.248]. Adding the response to V50 gives holding MAE delta
+0.97 s [-0.17,1.95]; passenger deltas cross zero. Joint holding MAE versus V48
is -79.08 s [-99.91,-57.23], essentially the travel-only gain, not an extra
turnover benefit. Raw holding coverage remains 58-67%; late arrival-clock MAE
is 101/86/95 s versus V48's 91/84/88 (development/reused/formerly-fresh).

Keep V50 as an endpoint-accuracy candidate and V48 as the timing reference;
do not make V51D the default forecast or change the real controller. These are
development forecast results, not actual policy gains. Next address late-window
clock propagation and path uncertainty, and align the holding surrogate with
the actual lower-policy observable inputs before further action-response use.

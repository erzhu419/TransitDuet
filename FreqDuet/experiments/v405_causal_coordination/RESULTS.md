# V405 Development Result

Subsequent unit audit found that this candidate's censored-state historical
OD prior divided hourly demand by 60 rather than 3600. The physical outcomes
below remain the paid as-run record, not a valid test of the intended forecast.
V406/source02 repairs the unit without replacing these records. No V405 restart.

Twenty starts were charged. Eighteen full paths passed physical validation;
two completed rollouts failed the dispatch-record/budget-auditor schema check.
No automatic traffic retry. Recording repair passed zero-traffic regression.
Incomplete paths are excluded, not replaced with cached scores.

Holding transfer minus the frozen legacy 10-s stop-cap controller, two paired
days per condition:

| Condition | Journey s/person | Actual CV2 | Readiness s/trip | Public delay s/day |
| --- | ---: | ---: | ---: | ---: |
| Nominal | +8.370 | -0.029613 | +34.844 | +9033.0 |
| Heavy | +24.915 | -0.024625 | +41.380 | +10748.5 |

Four holding-transfer paths are complete. They restore regularity but do not
improve the service/circulation tradeoff against the simple comparator.
The two validated joint paths equal their holding-only counterparts and already
violate the declared per-day delay/readiness guards. Thus the joint development
gate cannot pass regardless of the two missing paths. No confirmation or promotion.
The missing joint paths did publish plan changes; their effects are not inferred
from the two completed no-edit days.

Heavy-demand censored-model admission now works: the two holding-only paths
record 4,979/4,967 censored queries and 852/834 actual model overrides. Two-stop
proposals are not equivalent to delivery: 30/38 proposed, 10/6 followed, 20/32
abstained. The original physical fleet, seven legal actions and irreversible
command accounting passed on every admitted path.

Do not spend additional traffic merely to rescue a known-failed gate. The next
independent design aligns the model reference with the strong simple comparator
while retaining all seven candidate commands (V406); it does not loosen the
actual CV2/delay/readiness screen or tune on these demand days.

Evidence: review/readout_fixed/results.json and review/checks_recording/results.json.
All traffic and readout ran through scheduler high on CPU servers; only the compact
readout was pulled locally. Original code, historical results and main policy unchanged.

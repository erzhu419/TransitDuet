# Stage201: frozen wait-selection diagnosis

Explain Stage200 before changing the wait critic. Reuse all47 prediction files,
240 mean8 states and all three margins; no fitting, inference or native episodes.
Decompose each policy into identity, retained, cancelled and rerouted actions.
Cost/wait changes are divided by the entire evaluation scope, not subgroup size.
Inspect all five nonidentity candidates' raw wait residuals by root, action,
queue (zero,0-50,over50 passengers) and prefix headway change versus identity.
The50-passenger threshold is the existing native feature unit, not outcome-tuned.

Run: `native_transit_wait_selection_stage201_diagnostic_20261013_r1`.
One scheduler CPU task,3072MB, unbound node001-node006. Detailed state records,
targets and cached predictions stay server-only; only the summary is pulled.
Next: use the decomposition to decide whether conditional wait credit or
selection calibration is the justified change; no held-out margin retuning.
Limits: descriptive reused-label diagnosis, not independent confirmation,
CI-supported improvement or full learned-HRL validation.

# Stage259: Prefix Removal Influence

Saved Stage258 inner16 versus frozen full16, same old keys/reference/basis and
opposite16 futures. All eight variants checked; no fitting or selection.
t165559 completed on node006, exit 0, no retries, 65 seconds. Seven tests passed;
32 cells / 640 saved fits covered. Actions saved before evaluation panel access.
Stage258/255 controls reproduced within 3.47e-18; only 2,739,502 scalar bytes pulled.

Mixed expanded8 / pooled-safe removed group (12 bank-cells per family):

| Action Disagreement With Full | free_old | shrunk |
|---|---:|---:|
| Excluded prefix | 2.778% | 2.639% |
| Retained, same root | 0.764% | 1.007% |
| Retained, other root | 0.778% | 1.278% |
| Held, other root | 1.222% | 1.306% |

Excluded feasible cost-prediction changes are 7.93 / 7.47 times retained RMS:
local amplification and nonlocal action coupling coexist. Removing
401/914010400 leaves its own actions unchanged but changes held actions.
On held states, inner-minus-full mean
cost is +0.000077203 / +0.000043001 and wait +0.000135278 / +0.000176910 min.
Both pooled future halves worsen, while retained waiting improves.

Limitations: development/fit-exposed states, repeated held exposures, no seed CI.
Pooled signs differ across some banks/views. Removal changes fit and calibration
together; function change is not truth error or proof of required nonlinearity.
Stage253/248/169 closed; no deployment/default change.

Next: hold the 16 training futures and old keys fixed; compare prefix54 fits
with frozen versus re-estimated noise weights/shrinkage calibration. Separate
fit-population removal from recalibration before changing representation.

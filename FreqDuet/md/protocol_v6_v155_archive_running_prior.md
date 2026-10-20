# V155: frozen external-archive revalidation

Freeze V154 running moments5be4cea534 and V151 full eight-day service prior;
no refit or weight/gate tuning. V154 LODO gate remains unpassed. Reuse V145
archive:12 actors x32 disjoint days2451001--2451032 xold/handoff=768 paths.
V145 was consumed for interval audits;this is external-archive revalidation,
not brand-new confirmation. Do not use V151's16 consumed confirmation days.

Same V154 shadow formulas,complete-leg replacement only,same strict prefixes.
Native V151 full eight-day service reference on identical archived paths.
Unchanged requirements:all/hour2 full-MAE improvement;clearance/hour13/low-support
full-MAE non-harm;running-MAE non-harm all/hour2/clearance/hour13. Keep missing
support separate from unmet intervals;do not retroactively pass V154.

Scheduler,node001--006,8 workers;tests and first pair before remaining383 pairs.
Check archive fields and prior-day disjointness before scoring. No replay or
training;main,V146(3s,0s),3600s window,.1/.1 tails unchanged. Small summaries
only;all models,paths and cell data stay server. No controller admission.

Source107db12b04,scheduler t204492. Initial dispatch misclassified the negated
word "training" in its description;only the description was corrected. CPU
training permission remains disabled;scientific source and parameters unchanged.
Completed on node005,exit0,no retries;10 tests passed,384 pairs/768 paths.
Only95497-byte summary pulled. Paired full-ETA MAE delta(old/handoff,seconds):

| Scope | Old delta,95% CI | Handoff delta,95% CI |
| --- | --- | --- |
| all | -9.535 [-9.785,-9.282] | -9.538 [-9.787,-9.289] |
| hour2 | -11.960 [-12.867,-10.932] | -11.993 [-12.902,-10.970] |
| clearance | -0.463 [-0.969,0.028] | -0.454 [-0.962,0.033] |
| hour13 | -0.505 [-0.995,-0.039] | -0.503 [-0.999,-0.045] |

All12 actor and32 day margins improve in all/hour2. Clearance running MAE
improves1.112/1.102s with CI below0,but full-ETA non-harm remains unconfirmed.
All complete prior legs have8 days:low-support group is absent,not passed.
Original fixed_joint_gate=false remains. Freeze the useful candidate;next V156
only diagnoses clearance cancellation and fixed0/1-day running-history deletion.
Do not relax gates,retune weights,refit priors or switch main. This consumed
archive result is not fresh independent confirmation or control efficacy.

# V156: frozen clearance accounting and history-thinning stress

Same V14532-day archive/12 actors/both modes,768 existing paths. Freeze V154
running moments,weights n/(n+1),V151 eight-day service prior,main,V146(3s,0s),
3600s window,.1/.1 tails. No fit,replay,training or new confirmation days.

Fixed running-history arms:all eight days,earliest calibration day2751001 only,
and none. No best-day selection. First reproduce V155 full-prior endpoints and
verify zero-prior points equal V151,then process remaining383 pairs. Only prior
availability changes;service prior and every forecast formula remain fixed.

Clearance common structured support:partition true running/service/residence
errors and exact cancellation before episode/crossed averaging. Include opaque
endpoints only in full ETA metrics. Compare all/hour2/clearance/hour13 and observed
clearance launches,without redefining V155's failed joint requirement.

Single/zero histories are controlled support-deletion stress,not natural low
support or independent validation. Zero-prior equality is a fallback identity.
Report intervals and denominators;missing groups remain inconclusive. No gate
relaxation or main promotion. Scheduler,node001--006,8 workers;small summary only.

Source8d1e201a13,t204531,node006,8 workers,exit0,no retries.10 tests passed;
384 pairs/768 paths completed. V155 endpoints and zero-prior fallback reproduced.
Only343878-byte summary pulled. Paired full-ETA MAE versus V151,seconds:

| Running history | All old/handoff | Clearance old,95% CI | Clearance handoff,95% CI |
| --- | --- | --- | --- |
| eight days | -9.535/-9.538 | -0.463 [-0.969,0.028] | -0.454 [-0.962,0.033] |
| earliest day | -6.998/-6.999 | -0.990 [-1.476,-0.549] | -0.984 [-1.473,-0.549] |
| none | 0/0 | 0 [0,0] | 0 [0,0] |

On common structured clearance support(42858/42845 queries),eight-day running
MAE improves1.112/1.102s. Service/residence errors are unchanged;lost cancellation
adds0.648/0.648s,leaving full-MAE gains0.464/0.455s. Thus other errors do not
grow,but their previous cancellation with running errors weakens. Single-day
running gains1.275/1.275s lose0.282/0.288s to cancellation. Do not combine these
budgets with the slightly different full-endpoint cohort(42965/42959 queries).

Hour2 gains remain11.960/11.993s(eight days),8.016/8.036s(single day). Single-day
clearance launch MAE worsens10.865/2.329s,versus eight-day gains3.120/7.574s;
only3/4 queries,no crossed CI. Zero-day forecasts reproduce V151 per query,
including unsupported running-leg fallback;this is identity,not new validation.

Original V155 joint gate remains unpassed. Next summarize direct paired
single-minus-eight-day contrasts from saved cells and retain sparse launch
cells,without forecasting again,changing weights or promoting the single day.
Limitations:controlled deletion changes both prior composition and n/(n+1)
strength. This one fixed prior day cannot establish which effect explains its
clearance advantage,or validate natural low-support performance. Main,V151,V146,
window and tails remain unchanged;no new replay,fit or control experiment.

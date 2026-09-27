# Stage-24 Tail-Credit Decision Result

**All three frozen critics choose identical actions and hurt root 209061.**
Preflight `t101520` and full tasks `t101523/101524` completed at `b67004479e`;
full tasks ran on node006/node001. Each root retained 16 states and the 32/32
split. Retrieved 90,194 bytes of full results; zero new samples or fits.
Nineteen tests passed; independent cache recomputation matched the results.

Primary direction: futures 0-31 select oracle actions; 32-63 score them.
Positive benefit is mean local ISE reduction against observed short-window
credit alone. Each entry is mean benefit (conditional Monte Carlo SE), not CI.

| Root | Oracle tail | Each frozen critic | Switches: oracle / critic |
|---|---:|---:|---:|
| 209011 | +0.011023819 (0.001443545) | +0.007247420 (0.001438918) | 2/16 / 2/16 |
| 209061 | +0.001080515 (0.001022512) | -0.003889966 (0.001492252) | 2/16 / 2/16 |

On 209061, critics miss both oracle switches and add two harmful switches.
Its oracle gain is only about one conditional Monte Carlo SE.

Reverse-half sensitivity gives oracle benefits +0.007690870/+0.003003690
and critic benefits +0.006246429/-0.004360075 for roots 209011/209061.
Oracle choices agree across directions at all 32 states.

Next: freeze a cost-sensitive now/wait objective with a no-correction control,
online causal inputs and whole-path holdout, instead of expanding tail-MSE
capacity. Retain Stage-9 and the failed Stage-23 gate; no deployment follows.

## Limitations

References use counterfactual window costs unavailable online. These reused,
fixed-state contrasts are not policy gains or a certified oracle upper bound;
reverse splits are not independent evidence. Conditional Monte Carlo SE
does not establish cross-path or cross-root skill.

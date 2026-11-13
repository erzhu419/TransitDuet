# V413: Accepted Component Observer and Old-Day Readout

`t215008` passed on `2520208`: all 12 old streams, 5,200 random events,
endpoints and both public-slot records are exact (only query wall time is
excluded). All 522 slots are present; 37 are unsupported, and 5,818 vehicle/
scenario component records close to the unchanged forecast return clocks.

Read-only task `t215011` compares matched current supplying cycles. At 300 s,
the 3 missed late launches have total return error -411 s: travel -407,
service/PREPARE +5, holding residence -9. The 10 false alarms total +762 s:
travel +760, service +7, holding -5. These are one old day's diagnostics.

| Public trip, bus 10's current cycle | Travel error | Service/PREPARE | Holding residence | Return error |
| --- | ---: | ---: | ---: | ---: |
| 236, cycle 227 | -161 | +14 | 0 | -147 |
| 249, cycle 236 | +138 | +25 | 0 | +163 |
| 260, cycle 249 | -189 | +27 | 0 | -162 |

Values are predicted minus realized seconds. All three project and execute
zero further requested H; their residence is the same mandatory zero ticks.
Uniformly reducing H cannot remove these differences under the frozen rules.

At trip 225's advance sample (40200 s), modeled 212 returns 38 s before 214,
but actual 212 returns 19 s after 214. The -57 s relative error decomposes as
travel -6, service -6 and holding residence -45. Thus return-clock errors and
rank inversions need not have the same dominant component.

**Next.** Separate running extrapolation from reference-versus-actual holding
continuation before a method change; do not add a uniform ETA or rank guard.
The recorder is now physically verified, but these are descriptive residuals
on one consumed day, not causal attribution or independent control gains.
No additional paths, new days, training or policy changes were released.

[Parity receipt](review/component_parity/results.json),
[small component readout](review/component_readout/results.json).

# V412: Readiness Loss Under Slower Running

Server task t214968 read the 24 archived slow-setting paths, with zero
simulation starts. The 262 launches and returns per path, executed holding,
and both published delay endpoints reconcile. Public and eligible clocks
make no contribution to the paired delay difference on any of the eight days.

Against the randomized transition rule, V406's readiness delay is lower on
five days and higher on three (+99, +422, +4,382 s/day on 2520205/07/08).
It remains lower against pure fixed10 on all eight days (including -3,418
s/day on 2520208). The reference matters; the original joint gate stays failed.

On 2520208, 92 departures become later and 119 earlier relative to the
transition rule; the positive and negative sums are +6,895 and -2,513 s.
Hours 10--13 after service start account for +4,247 of the +4,382 net
seconds. The 13 trips with a different supplying predecessor account for
+1,504 s; 237 common-predecessor trips account for the other +2,878 s:
+1,844 inherited departure, +780 holding, +528 other cycle time and
-274 terminal interval. These are sums over linked trip observations,
not independent delay sources or an attribution to particular actions.

The largest sequence is physical bus 10, trips 236 -> 249 -> 260, with
+734, +584 and +659 s of delay; all three leave one tick after their
respective returns in both arms. Trip 236 uses different predecessor
labels (227 versus 225), so its source cannot be split by the common-trip
identity. On 249 and 260 the prior trip holds 150 and 160 s less; inherited
lateness persists, and 260 adds 235 s of non-holding cycle time.

**Follow-up.** The [supply and forecast audit](SUPPLY_FORECAST.md) reconstructs
the FIFO reassignment on 2520208 and compares archived resource clocks against
realized clocks across all eight slow days. The capped controller and gate are
unchanged.

[Small circulation summary](review/circulation/results.json)

# V412 slow-speed supply and forecast audit

The read-only audit reconciled all 16 archived slow-speed paths (eight demand
days, capped V406 and randomized transition). Every actual launch used an
eligible earliest-return vehicle. No traffic paths were started. The original
slow-speed joint gate remains failed.

On day `2520208`, the first supply reassignment occurs at trip 206. The later
bus 10 sequence is explained by the order of returns from trips 212 and 214:

| Clock (s) | Capped V406 | Randomized transition |
| --- | ---: | ---: |
| Bus 10: trip 212 launch / return | 38798 / 40942 | 38250 / 40255 |
| Bus 11: trip 214 return | 40923 | 40947 |
| Trips 225 / 227 assigned vehicles | 11 / 10 | 10 / 11 |
| Trip 236 eligibility / actual launch | 42480 / 43588 | 42480 / 42854 |

Bus 10's trip 212 was already 548 s late relative to the comparison arm and
took another 139 s longer. Bus 11 returned 19 s before bus 10 in the capped arm,
so FIFO assigned trip 225 to bus 11 and trip 227 to bus 10. Trip 236 then used
bus 10 one tick after its return in both arms, with different predecessors.
This explains the reassignment, not the cause of the earlier timing changes.

At query 42469, the first recorded model scenario forecast bus 10's trip 227
return and next launch at 43398/43399, both 189 s early. The selected and
model-native forecasts were identical for this bus. A relative resource check
can therefore miss a shared clock error. It does not establish that this query
caused the later delay. The archive records next-launch clocks, not predicted
supplying trip IDs; the audit matches realized clocks by vehicle and trip only.

Across the eight days, next-launch MAE was 83--102 s. Mean signed error was
+5.2 s on the largest readiness-loss day, -3.0 and -8.4 s on the other loss
days, and -25.0 s on the largest readiness-gain day. These are repeated
vehicle-query forecasts, not independent observations. A uniform ETA
correction is not supported. A future method test should instead ask whether
query-time return order and the gap to the next eligible departure identify
fragile supply before it changes; realized supplying IDs are labels only.

[Small audit summary](review/supply_forecast/results.json)

[Follow-up return-order and public-slot pressure screen](SUPPLY_ORDER.md)

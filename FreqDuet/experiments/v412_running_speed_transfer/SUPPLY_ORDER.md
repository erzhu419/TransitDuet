# V412 return-order and public-slot pressure screen

Read-only audit of the eight slow days, using accepted-override receipts and
the first archived scenario. For each direction's next outstanding public
slot, take the latest receipt in the preceding 300 s. Reconstruct the supply
set from strict prior dispatch and return events; exclude ambiguous same-tick
events and the initial pool. Score only when no compatible bus is ready and
every compatible inbound vehicle has a return forecast. Realized return
order and cross-arm predecessor changes are evaluation labels only.

- 711 of 2,096 slots had a snapshot; 169 already had a bus ready and 542 had
  complete inbound forecasts with none ready. The first forecast return was
  correct on all 542, but **all** forecast first-to-second gaps exceeded 30 s.
  These snapshots covered only 4 of 41 cross-arm predecessor changes, and
  none of the 13 changes on day `2520208`.
- On the 542 scored slots, the first-return clock identified 467 of 475
  shortages by public time, with 8 misses and 12 false alarms. This is
  522/542 correct, versus 475/542 for always predicting a shortage on the
  same subset. It is not an all-day alert evaluation.
- On `2520208`, trip 225 was due at 40500 s. In all 10 same-query forecasts
  before then, both model continuations ranked trip 212 ahead of 214. The
  last pre-public forecast (40143 s) put 212 ahead by 51 s (selected) and 56 s
  (native). In the actual capped path, 214 returned 19 s first. The last
  paired receipt (40870 s) recovered the direction, after the public slot.

The clock has conditional value for short-term supply pressure. The archive
does not validate a near-tie trigger, and the focal error was confidently
outside the proposed 30 s band. Do not turn this into a holding gate. Any
next test of supply uncertainty needs observations indexed by public slots,
including no-action states, and an independent day set. The V412 slow-speed
gate and the main policy are unchanged.

[Small audit summary](review/supply_order/results.json)

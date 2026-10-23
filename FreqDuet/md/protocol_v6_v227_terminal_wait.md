# V227: read-only terminal waiting and dispatch clocks

Use only V225/V224 native records and V226 totals. No replay,training,parameter
change or policy switch. scheduler high/manual,node001--006;small summary only.

Rebuild posted plans in native callback/dispatch-loop order,not just final plan.
Rebuild physical terminal inventory from the first complete snapshot and native
return/launch events;check snapshots,FIFO choices and final launch plans. Frozen
runtime dispatches BEFORE driving:returns become usable at the NEXT tick.
Resolve the actual inherited config;the native terminal headway floor is off.

Partition first-eligible-to-launch delay into later plan blocks,no-ready-fleet
ticks and open/ready residual. Partition same-bus terminal idle into return-tick
boundary,posted-plan blocks and plan-open idle. Keep initial stock separate.
Retain overlap between plan blocking and no supply,and group idle changes by
earlier/same/later return and launch clocks rather than calling them free slack.
Also separate final posted plan,rounding and final-plan-to-actual gap against
the public timetable. Reproduce V226 readiness,public-delay and idle deltas.

Keep detailed trips remote. These are observed-path clock identities;neither
idle nor plan-open residual is automatic recoverable slack. No alternative
dispatch or passenger counterfactual is evaluated or selected on these days.

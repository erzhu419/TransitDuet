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

Source4ae335a897,t211276(high/manual,node006),exit0:9 tests and4 contexts passed.
All posted plans,88 fresh inventory snapshots/path,FIFO choices and V226 totals
reproduce. In all12 paths,the entire recorded readiness delay is no-ready-fleet
time;post-eligible plan blocks and open/ready residual are both0. Terminal idle
contains only plan-blocked time and the native next-tick return boundary.
Upper-V107 readiness increments75/221/83/221s are therefore genuine supply waits.
On day2151002,both actors have33 late same-bus launches:+249s launch and+249s
inbound-return difference,0 idle change. Extra idle primarily accompanies other
earlier returns with unchanged departure,not the late-launch chains.
Next:locate the FIRST new inbound return delay,split inherited launch lag from
holding and other elapsed,and then inspect observed service/run components.
Do not cut legitimate pre-plan idle or claim spare slack. No new control result,
replay,training or main switch. Only the42749-byte summary returned locally.

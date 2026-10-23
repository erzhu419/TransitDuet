# V241: V240 passive request integration and synthetic no-op preflight

Append an independent FIFTH sidecar after frozen V235's completed cycle query.
Use its already updated view and V230 native frontier;do not advance CycleState
again or resample native F/B,positions,plans or holding. Preserve old prepare
clock,query occurrence and both native/sidecar ordinal domains.

Record exact frozen V238 and V240 supply-blind requests plus H=0 aliases.
Inherit the frozen native __call__ without overriding it;all requests remain
unexecuted. Only V107 and frozen upper engine arms are accepted. V240 upper_only
maps explicitly to old upper,not old joint/V219 lower. New recording counters
and RNG/native-call guards belong to the new sidecar,never old observers.

The acceptance routine compares complete legacy,V224,V230 AND V235 payloads
recursively and exactly,then recomputes every new packet from those unchanged
old sidecars and the frozen request functions. Missing/duplicate joins,altered
clocks/requests,extra future fields,old counter changes or executed shadows fail.
Actual H persists through replans and closes/opens only through the old observed
physical-cycle transitions. Shadow requests never spend/refund holding.

Current scope:synthetic callback producers only,no simulator/model imports,
archive query evaluation,passive replay,smoke,formal future or training. The test
entrypoint has no collection command. All104 V240 reserved starts stay unspent.
Scheduler high/manual,node001--006;small JSON/log locally,all details remote.

Next gate:separately authorize passive_c0_v107 against accepted V236 originals;
only a full equality/recording pass permits passive_c0_upper_only. Synthetic
success is implementation evidence,not empirical path equality or control gain.

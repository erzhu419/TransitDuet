# V226: read-only V225 coordination diagnosis

Only4 consumed V225 joint and matching V224 V107/upper archives. No replay,
training,parameter change or policy switch. scheduler high/manual,node001--006.

Compare joint-upper,upper-V107,joint-V107. Match trip/stop/direction/occurrence;
separate phase override,native state response,forward/follower/branch changes,
clipping,quantization and feasibility. Close against actual trip holding and
V225's saved endpoints;preserve unmatched/repeated decisions explicitly.

Reconstruct actual arrival CV2 from launch,holding,other elapsed and covariance
terms,including mean normalization. Join returns to the NEXT launch on the same
physical bus using saved native event order. Decompose matched inbound chains
into prior launch,holding,other elapsed and terminal idle;initial stock and
changed assignments remain separate. No imagined slack or unrun action labels.

All detailed decisions and supply chains stay remote;return only small totals.
These are trajectory identities,not independent causal effects or efficacy.
Passenger evidence is limited to existing endpoint waiting/onboard exposure.
Decision:locate the observed conflict before designing a separately registered
candidate;do not tune gains or reuse these days as independent confirmation.

Source ec0e6616d7,t211269(high/manual,node006),exit0:8 tests and all4 contexts
passed accounting;V225 endpoints reproduced. Each pair matches5200 decisions,
262 physical launches and250 inbound chains plus12 initial-stock launches;
no matched-role identity or feasibility-set change. Continuous phase corrections
of0.125--0.25s caused5--15s bin changes. Joint-upper direct/native/net holding
seconds are(-15,+60,+45),(-5,+10,+5),(-15,-10,-25),(-5,+15,+10).
Upper-V107 already increases CV2/readiness in all4 and total public delay by
89/221/83/221s. Day2's upper-only supply decomposition includes+141s propagated
parent-launch differences and+78/+75s terminal idle,not just extra holding.
Next:read-only distinguish published-plan waiting from supply/eligibility timing
in the existing same-bus chains before redesigning upper coordination. Increased
idle is not automatically avoidable slack. No gain changes,new replay or training.
Only the45787-byte compact summary returned;all row-level details stay remote.

# V249 real passive equality preflight

Executes only the two V247 slots with frozen V248 `b83f1c5db4` instrumentation.
Actor 12037, demand day 2151001, sampler 402112137; original fleet and authority.
Reuse V242 `c6710744f9` full saved references; no new reference paths.

First run server-side entrypoint/slot tests. Dispatch c0/V107 alone; only after
all complete-payload and recording gates pass may c0/upper be dispatched.
Any started attempt consumes its slot. No automatic retry, context replacement,
extra starts, intervention, training, selector definition or main-policy change.

The old five payloads and independent PREPARE timing sidecar stay on the server.
Only small checks/status and failure logs are returned locally.

Source `96960db57a`: entrypoint task `t211401` passed 4 slot/interface tests.
c0/V107 `t211402` and conditional c0/upper `t211405` both passed (exit 0),
on node004 via scheduler high/manual dispatch. Each preserves all five original
payloads exactly: actions, clocks, endpoints and old sidecars. Each retains
5,240 PREPAREs, 5,200 queries, 40 unused PREPAREs and 2 zero-query cycles;
5,200 fronts have complete timing and 4,940 same-cycle/front transitions close.
There are no open cycles or missing query joins. Budget: 2/2 used, no retry,
extra reference run, intervention or training. V238 and main are unchanged.

Next: a read-only V246 feasibility audit of these saved paths, separating local
compensation risk from current due-service exposure before any rule is defined.

## Limitations
Neither path had a PREPARE-to-QUERY H increase; delayed-H separation is
covered synthetically, not exercised here. No extra path is opened for it.
Passive equality is recording evidence, not a control-performance improvement.

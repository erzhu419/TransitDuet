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
c0/V107 task `t211402` passed (exit 0): all five old payloads exact;
5,240 PREPAREs, 5,200 queries, 40 unused PREPAREs and 2 zero-query cycles retained.
Second slot c0/upper is now being dispatched. V238 and main are unchanged.

## Limitations
The first path had no PREPARE-to-QUERY H increase; delayed-H separation is
covered synthetically, not exercised here. No extra path is opened for it.
Passive equality is recording evidence, not a control-performance improvement.

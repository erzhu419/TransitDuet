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

Status: integration and real preflight pending. V238 and main are unchanged.

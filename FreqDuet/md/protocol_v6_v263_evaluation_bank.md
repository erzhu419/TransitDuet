# V263: E collector, synthetic admission only

Frozen V253/V254 and S seal65bdd96938. Bind only the32 E units and their fixed
A/B halves; preserve the exact S0 pre-command prefix, then fork the registered
E root/current window using the accepted V257/V259 observer and execution chain.
S code and artifacts are read-only. No monkeypatch of S phase/root validation.

Native first, then ALL legal nonalias bins, even when the S-locked primary is
native. Preserve masks and all outcomes; never choose a new E winner. Carry
the sealed contrast ID with each path and unit. Actual H/first-next-supply
accounting stays unchanged. No online selector or policy change.

Real E requires a separate source/S-lock-bound evaluation_authorization.json,
at most7 starts/unit,224 total and8 workers. Failures burn slots,no automatic
retry or replacement. This stage writes no real authorization and starts no E.
Server scheduler/high/manual synthetic tests only; return a small receipt.

Accepted source e414070e27,t211535,node004/high/manual,backend exit0:
15 tests passed,0 failures/errors. All32 E bindings and224-slot synthetic
accounting passed; S choices/seal unchanged,real E starts0,authorization absent.
Earlier t211533 exposed reading a missing plan after native-load failure; the
status check now precedes plan access. Its failed synthetic record is retained.
Next: separate explicit real E authorization; no S/E reselection or policy change.

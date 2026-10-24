# V247 PREPARE timing mapping and minimal passive preflight

V246 source: `1828d9cd02`. This stage seals a field mapping and budget, not a
runtime recorder or action selector.

## Mapping

- At the existing native PREPARE stamp, read own executed H from the actual bus
  counter minus its actual trip-launch baseline. Copy identity, physical cycle,
  launch time and sample clock. QUERY H remains a separate later observation.
- At that same stamp, copy the already completed front departure timing and
  stored launch record. Join the original V224 prepare occurrence afterward;
  match bus/trip/station/direction, launch and departure times and old orders.
  Missing front timing remains unavailable, never zero or backfilled.
- Native event seq and V224 sidecar order are distinct domains. For supported
  rows, verify `F = launch gap + executed-H gap + nonholding elapsed gap`.
  Keep all PREPAREs, including unused ones, and every query occurrence.

## Reserved Preflight

Two fixed sequential starts: c0/V107, then c0/upper-only **only after the first
passes all gates**. Actor `12037`, demand day `2151001`, original sampler and
native paths; reuse V242 source `c6710744f9` references without new reference runs.
Failures consume slots; no automatic retry, context replacement or expansion.
Reservation does not authorize starting either path.

Acceptance requires recursive exact equality of the entire legacy payload and
V224/V230/V235/V241 sidecars, including actions, clocks, endpoints, old counters
and shadow requests. Add a sixth independent payload only. Every forward-valid
PREPARE needs complete matched components, with unchanged RNG/native-call counts.
Any mismatch stops progression; no tolerance or ignored fields.

## Status

Server synthetic/source checks pending. This turn starts zero real paths and
performs no archive scan, training, fitting or online selection. V238, main,
fleet, seven-bin authority and all other frozen components remain unchanged.

Next: implement the independent passive recorder and run synthetic checks;
real-path execution needs separate approval. Runtime equivalence is not yet tested.

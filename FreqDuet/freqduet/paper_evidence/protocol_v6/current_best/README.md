# Protocol V6 Current-Best Source Evidence

This directory contains the small immutable result artifacts used to build the
current FreqDuet paper tables. It does not contain checkpoints or full training
logs.

## Included Experiments

- `v8_confirmation/`: successful independent 40-episode confirmation of
  `F_freqduet_protocol_v6_avlcompact_w2_hiro` against the matched no-guard
  controller. The paper alias is
  `F_freqduet_protocol_v6_confirmed_main_hiro`.
- `v9_longtrain/`: independent 200-episode evaluation of that alias. Its
  registered status is `longtrain_not_confirmed` and must remain visible.
- `v9_external/`: source-identical V9 comparison with fixed headway, rule
  holding, and rule MPC.
- `v33_timescale/`: preregistered 200-episode development screen of lower-actor
  freezes at episodes 40, 80, and 100. All three candidates failed the locked
  headway-CV gate, so confirmation was not authorized. This is valid negative
  development evidence, not a new controller claim.

Build the normalized draft evidence package with
`scripts/build_freqduet_protocol_v6_evidence_package.py`. The builder rejects
artifact, protocol, source/scenario, comparator-roster, V33 no-pass, or
config-fingerprint mismatches. The generated package is intentionally marked
`submission_ready: false` because the V9 long-training gate failed.

The historical V1/composite package and V28-V33 development failures are not
part of the current controller claim. They remain available only as historical
or negative-result evidence. In particular, the favorable unfrozen-control row
inside the V33 development roster cannot replace the failed independent V9
long-training gate.

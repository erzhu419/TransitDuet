# MuJoCo v14.29 fresh-anchor bank outcome

## Execution identity

- Frozen algorithm revision: `fc7fa8d8c1e55325af9cb32efece3e0cfc2bbd3c`
- Freq-HRL source manifest: `02f3ba95376021dff0aa11f30d46dd6159e63b55a1d2678d6011ea350745af39`
- Frozen launcher/protocol commit: `8082a9008e`
- Run: `mujoco_v14_29_fresh_anchor_bank_20260830_r1`
- Scheduler tasks: `t84875..t84922`

The 48 one-core tasks were submitted through `scheduleurm` with
`allowed_nodes=node001..node006` and `require_node=None`. Dynamic placement put
all cells on node005 because it had sufficient free capacity. All tasks reached
`done`; none failed or were cancelled.

## Qualification

The frozen qualification analyzer reported:

- status: `fresh_anchor_bank_qualified`
- qualified anchors: `48/48`
- environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5
- optimizer seeds: 16 fresh seeds per environment

Every accepted cell matched the frozen revision and source manifest, exact
training/selection/evaluation seed registries, disturbance panels, router
contract, checkpoint-selection contract, and minimum selected iteration. The
serialized checkpoint digest also matched its summary, and every selected
checkpoint had an eligible finite selection score.

## Replicate diagnostics

Each environment produced 16 distinct parameter digests and 16 distinct
serialized checkpoint digests. Selected checkpoint iterations were 31--63 for
HalfCheetah-v5, 23--63 for Hopper-v5, and 15--51 for Walker2d-v5. Thus there is
no observed duplicate-model failure that would invalidate optimizer seed as the
replication unit.

This result qualifies the anchor bank for the preregistered v14.29 portfolio
confirmation. It is not itself evidence that the restoration portfolio passes
the held-out confirmation gate.

# PointMaze Exogenous-Control Stage-5 V1 Preflight

Date: 2026-09-21

## Outcome

The two registered preflight tasks completed:

- `t99960`: flat history on `node004`;
- `t99961`: hierarchical history on `node006`.

Only their compact `result.json` files were synchronized. Both tasks used the
frozen algorithm revision `df516684f8ea2fbb89fb65fa038e11a44a005200` and the
registered runtime stack.

## Audit

The preflight passed the software gate:

- both methods completed two training iterations and finite PPO updates;
- upper, lower, and flat states were all 134-dimensional;
- flat PPO had 67,973 trainable parameters and HRL had 68,424, a ratio of
  1.0066 to the flat budget;
- each 64-step row was protocol-valid and preserved current physical feedback;
- current external values were visible before action, future values were not,
  and the stream was declared action-independent;
- measured force RMS was 0.12 on each axis with the frozen 0.04-second period;
- HRL produced three upper decisions, three lower option-credit boundaries,
  and nonzero upper and lower optimizer steps;
- untrained and final evaluations used the same held-out seed within method;
- the two cells were dynamically placed on different eligible CPU nodes.

The short 64-step rows have a strong start-near-target effect, including high
untrained tracking success. They are not performance evidence and are not used
for the registered learning gate. The fixed 300-step development protocol and
its paired untrained controls remain unchanged.

## Decision

The preflight authorizes the frozen 16-cell development matrix. It does not
support ordinary-HRL learning, frequency routing, or a Freq-HRL performance
claim.


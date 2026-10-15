# V59: finite-day system-credit learning smoke

## Contract

V58 passed once-only accounting without learning. V59 connects an isolated
categorical learner; the published main and frozen runtime remain unchanged.
Reuse the four V58 development days (two training seeds by two demand seeds).
This is a learning/interface smoke, not a policy-effectiveness confirmation.

- Holding remains `{0,5,10,15,20,30,45}` seconds; fleet and upper actor are fixed.
- Start from B200 deployment, preserve the old actor projection and add a
  zero-initialized projection for new features, and reset critics
  and optimizer state because the old critics have different targets.
- Actor inputs are existing causal local features, current identity/time,
  published target, current terminal readiness, completed AVL headway moments,
  and full per-bus pending command slots including presence and command order.
- Reward is negative once-only waiting plus onboard exposure. Units are fixed
  at 1,000 passenger-hours; this is a unit conversion, not a fitted weight.
  Retain LF/HF exposure in the accounting; no latent passenger queue or future
  return enters the actor state.
- Global replay links consecutive chronological decisions across vehicles and
  trips. Gamma is one; only the fixed-day endpoint is terminal. No TAP bonus,
  per-trip replay insertion, upper optimization, or duplicated passenger cost.
- Independent constraint: final **arrival headway CV squared**, matching the
  reported service metric, no greater than the matched frozen reference.
  Use the unrounded CV from the exact reference replay, not the four-decimal
  headline CSV value.
  Replay cost is the signed change of observed CV squared between decisions;
  its sum plus the uncontrolled prefix equals final CV squared. A decrease is
  valid and must not be clipped to zero. The multiplier updates once per full
  rollout from the full-day constraint gap, not the average replay cost.
- Reuse RE-SAC categorical networks, aggregation and optimizers. Finite-day
  reward/cost targets are not clipped. Fixed entropy alpha is `1e-5`; the old
  adaptive per-stop temperature cannot be transferred to this reward scale.

## Smoke

Each case performs a deterministic warm-start day, requiring bit-exact initial
categorical probabilities and exact original
holding path and episode metrics, followed by 32 SAC updates. Then run one
stochastic collection day and another 32 updates. Keep batch size and neural
network learning rates from the frozen configuration. Check conservation,
one day terminal, cross-trip/pending state, finite losses, actual actor/critic
updates, frozen upper weights, and seven-bin physical execution.

Run unit tests and the first case through scheduler before dispatching the
other three. Outputs/checkpoints stay on the server; retrieve only small status
JSON. No automatic main-policy replacement or 40/200-episode campaign.

## Limitations / Next Step

Full causal system state is not proven Markov; the raw global objective is not
yet an identified HF marginal advantage. The CV constraint is an expected
training-day constraint, not a hard per-day safety guarantee. Two collection
days cannot establish service quality, convergence, or superiority. Next reuse
server-side post-64-update actors for frozen/updated by deterministic/stochastic
matched rollouts, with identical deployment background and paired sampling
seeds. Separate learning from sampling before constraint repair and long training.

## Outcome (2026-10-15)

Source `fcae5fb63f`; scheduler `t162368`, `t162370`--`t162372` and aggregate
`t162373` (aggregation source `eeeccba52c`), all exit zero. Eleven unit tests,
four bit-exact warm-start days, eight full-day rollouts, 256 updates, batch 512,
41,600 replay transitions, eight day terminals, 41,551 cross-trip links, and
1,983 same-clock links. Maximum conservation error: `4.66e-10`. Main unchanged.
Small summary: `protocol_v6_v59_system_learning_split/summary/status.json`.

All four stochastic collection days violate the reference CV budget, despite
lower observed journey, holding and readiness delay. The deterministic reference
and stochastic collection are different deployment modes, so this is not an
effectiveness comparison. No 40/200-episode campaign was launched.
Prototype failures remain in scheduler/source history; fixes covered protocol
initialization, CSV/rounding boundaries, and bit-exact actor input extension.

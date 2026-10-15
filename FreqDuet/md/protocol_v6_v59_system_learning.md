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
days cannot establish service quality, convergence, or superiority. After smoke,
choose a small matched pilot with identical state/initialization and separate
credit/constraint controls before new-seed confirmation or long training.

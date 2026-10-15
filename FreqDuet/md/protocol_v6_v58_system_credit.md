# V58: finite-day system credit contract

Build a shadow lower SMDP stream before changing learning. Replay the unchanged
U200/L200/B200 main on training seeds 12011/933013 and demand seeds 45007/934011:
four days, reusing the first passed smoke. Require exact V56 reference episodes,
complete commanded-action paths and total wait/onboard/backlog exposure.

Charge each physical interval once along the chronological lower decision stream.
Keep the pre-first-decision uncontrolled prefix separate. Retain total and frozen
LF passenger exposure; HF is their difference. Include censored waiting after
service clearance until the original evaluation horizon. Vehicle returns never
close this system stream. Use undiscounted finite-day continuation (discount one
except at the true day terminal) and passenger-hours as a fixed reward unit.

Same-clock decisions are serialized in the original runner order. Their states
retain the complete list of earlier chosen, not-yet-executed holding commands;
zero-time links receive zero elapsed cost. Require proposed commands to match
physical execution. Preserve the existing encoded actor observation and published
target, adding only current terminal-ready counts and known pending commands.
Global queues, realized future returns and future outcomes are reward/audit labels,
not state inputs. No LF forecast is added to the lower actor.

Unit tests and a single end-to-end smoke precede the remaining three replays,
all through scheduler on node001-node006. Only small summaries/markers return.
Raw transitions and checkpoints remain on the server. The next learning step
must encode the full pending-command state and specify regularity constraints
without reintroducing duplicated interval cost or per-trip termination.

## Limitations

This is a verified credit/transition candidate, not a trained controller or
policy improvement. Raw shared passenger cost is not yet a validated HF marginal
advantage; LF/HF ownership is retained for that subsequent design. Old upper and
frequency modules, actions and fleet remain unchanged, and no shadow transitions
are inserted into the real replay buffer during this stage.

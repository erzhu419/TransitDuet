# V58: finite-day system credit contract

Shadow the frozen U200/L200/B200 main on training seeds 12011/933013 and demand
seeds 45007/934011. Charge physical intervals once, separating the uncontrolled
prefix before the first lower decision. Retain total/LF exposure (HF is their
difference) and censored waiting to the original horizon. Use passenger-hours
and finite-day discount one; vehicle returns do not terminate the system chain.

Same-clock states retain every earlier chosen, unexecuted holding command in
original decision order. Zero-time links have zero cost. Check proposed versus
executed commands. State uses existing encoded observations, published target,
current terminal-ready counts and known pending commands; global queues and
future return/outcome labels stay outside state. No LF forecast is added.

## Execution

Source `c23f414db6`; smoke `t162247` passed 11 tests and the first day. Batches
`t162259-t162262` reused that day and replayed the other three on Linux CPU nodes;
aggregation `t162263` completed with exit code zero. Across 20800 decisions:
all four reference episodes, complete holding paths and total exposures match
exactly; maximum credit conservation error is 4.6566e-10 person/trip-seconds.
There are 997 same-clock and 1022 return-crossing bootstrap links. The old local
stream has 1040 trip terminals; the shadow system stream has four day terminals.
Only small markers/status JSON returned; raw transitions and checkpoints remain
on the server. No real replay buffer or policy was changed.

## Next

Encode the complete pending-command state per physical vehicle, including a
presence mask so zero hold differs from no command. Connect the learner to the
once-only system stream and specify an independent regularity constraint: the
current two-sided incremental regularity is a reward bonus, not a cost channel.
Verify a small learning smoke before matched 40/200 trajectories and fresh seeds.

## Limitations

This validates accounting and transition semantics, not a trained controller or
policy gain. Raw shared passenger cost is not yet a validated HF marginal
advantage; preserved LF/HF ownership is available for that subsequent design.

# Stage-13 Deployed-State Timing Diagnostic

This is a **mechanism diagnostic**, not a performance screen or confirmation.
It uses only the revealed Stage-12 roots `209011/209061` and their saved
compact results. The Stage-12 controller is retrained with the same frozen
options; its selected checkpoint and each factual episode ISE/return must
match the saved rows exactly or the diagnostic stops.

For every held-out path, a seed-fixed draw selects up to two early-planning
bins and two deadline-planning bins, excluding the initial and final bins.
Each pair replays the actual deployed decision schedule and a schedule that
changes only the selected bin: actual early call versus offset-25 deadline, or
actual deadline versus offset-0 call. Both arms retain one upper call per bin,
have an identical causal prefix at the check, and run complete episodes on
the same exogenous path. All later planning times are frozen to the factual
schedule. We record the Stage-12 score, 50-step ISE advantage, and full-episode
ISE advantage; paired branch steps count in compute cost.

This identifies whether the frozen score's early/deadline decisions are
locally aligned at **states visited by the deployed policy**, under a static
continuation. It cannot measure the value of fully adaptive continuation or
authorize another threshold search. Run one preflight root `208001` first,
then the two revealed roots only if replay checks pass. Synchronize only
compact `result.json` from dynamically placed node001-node006 tasks.

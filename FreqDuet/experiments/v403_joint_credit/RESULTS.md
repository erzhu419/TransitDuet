# V403 completed: close the joint-credit candidate

**The frozen four-day development screen failed against both simple controls.**
Regularity improves, but passenger journey and dispatch delay worsen on every
day relative to both fixed mixtures. No independent confirmation or promotion.

## Completion

All 27 charged traffic paths completed: 3 engineering and 24 development paths.
There were no failed traffic starts or replacements. Eight zero-traffic checks
passed after repairing a missing base-class field in the synthetic manager
fixture. The original failed check t214179 is retained; t214181 passed. Neither
constructed a native Runner or used a traffic slot.

Engineering pressure and candidate-noop reproduced archived V402 actions,
timing, endpoints, slow records and random events exactly, with zero diagnostic
roundoff. The candidate pilot then admitted 9 genuine requests and closed its
credit ledger. Development used four new demand seeds 2430001--2430004, unchanged
V402 mixture weights and six physical arms per day. All physical/credit checks
and 5,200-event within-day pairings passed. No learning or main-policy change.

## Actual Development Effects

Candidate minus comparator; negative means lower. These are four-day averages,
not independent confirmation. Full descriptive bootstrap intervals and all day
effects remain in review/aggregate/results.json.

| Endpoint | Static mixture | Stop-cap mixture | Old pressure |
| --- | ---: | ---: | ---: |
| Journey, s/passenger | +6.803 | +10.744 | +12.768 |
| Waiting, s/passenger | +1.672 | +2.614 | +3.037 |
| Onboard, s/passenger | +5.131 | +8.130 | +9.732 |
| Public dispatch delay, s/day | +5917.10 | +6165.45 | +9918.50 |
| Readiness delay, s/trip | +22.793 | +23.764 | +38.211 |
| Actual CV2 | -.013690 | -.012680 | -.020795 |
| Executed holding, s/trip | +22.333 | +24.454 | +26.250 |

Both mean-delay predicates fail; both mean-CV2 protections pass. The CV2 reduction
is real, but does not rescue the joint gate. Passenger journey and public delay
are worse on 4/4 days against each simple mixture, not just one outlying day.

## Interpretation And Disposition

The new decision changes actual control: 985 evaluated requests yield 34
admissions, 919 no-joint-improvement results and 32 unsupported references.
Most proposed restrictions are therefore not imposed; actual holding rises to
146.594 s/trip, versus 120.344 for old pressure. Protecting the common reference
restores regularity but loses the operating savings. This version moves the
operating tradeoff; it does not improve the demonstrated tradeoff frontier.

Close this candidate as preregistered. Do not relax its constraints, sweep
retention, repair forecasts on these days, start confirmation, or promote it.
Keep V402's turnover benefit and V403's regularity result as paired evidence of
the unresolved service--turnover tradeoff, not a learned-frequency advantage.

## Limits And Reproduction

The reference includes existing admitted credit and uses forecast native
continuation, not an exact future closed-loop guarantee. Saved rejection counts
do not isolate how much each constraint caused the full-day loss. Four development
days do not prove all resource-allocation designs ineffective.

Frozen source commit 1da9b1d6da; physical worker source03 under the dedicated
FreqDuet-v403-joint-credit server root. Scheduler high/manual, node002--006,
24 concurrent development paths. Median/max test path time 24.36/27.27 minutes.
Raw trajectories remain server-side; only compact summaries were synchronized.
Traffic tasks: t214187/t214188/t214194 and t214199--t214222; aggregate t214232.

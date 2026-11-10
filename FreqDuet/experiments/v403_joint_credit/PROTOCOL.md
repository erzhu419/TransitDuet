# V403: joint credit and holding admission

Freeze manual joint, K4, forecasts, waiting-plus-onboard model cost, native
continuation, actual nonterminal arrival CV2, 12 vehicles and seven legal bins.
No learner updates, main-policy change, synthetic slack or retrospective edits
to submitted commands. V402 and its failed gates remain unchanged.

The original rho=.50 pressure formula now PROPOSES a remaining-trip allowance
at each original 7200+900k publication. It does not immediately constrain a bus
or enter the plan supply forecast. The plan algorithm sees only admitted credit.
Requests expire at the next publication or physical return. Actual commands
between proposal and admission debit proposed remaining credit as well; spent
commands cannot be retroactively squeezed into a later allowance.

At the proposing physical cycle's next supported native holding query, first
run the unchanged fast selector with existing admitted credit. Its selected
continuation is the COMMON no-new-tightening reference. Enumerate the original
legal current commands under the proposed tighter credit, including its native
continuation. Require the unchanged graph feasibility test against the COMMON
reference (CV2, returns, next launches and dispatch delays) and non-higher
waiting-plus-onboard cost. Minimize that cost; exact cost ties prefer lower
predicted total public delay. Admit only a strict cost or delay improvement.
Publish the chosen credit before irreversibly submitting/debiting its current
command. Existing credit is never increased or reset. No supported evidence
means no new tightening, not removal of already admitted constraints.

This co-decides one physical cycle's credit and its current command. Other
cycles, published timetables, pending commands and strictly completed histories
are identical within the comparison. Future native continuation is a forecast,
not a guarantee that future replanning reproduces its path. Only actual full-day
outcomes decide scientific benefit.

Budget: at most 27 charged starts, no automatic replacement or retry:
1. pressure on old day2410001 must reproduce V402's archived reference;
2. the new selector with the unchanged pressure manager/no proposals must also
   reproduce that reference exactly;
3. joint credit on the same old day validates real admission/accounting only.
The first two gates and zero-traffic checks precede path3. Path3 must complete
with valid accounting and at least one joint credit admission before the screen.
Its outcomes do not tune any parameter or authorize more engineering starts.

Screen: new simulation demand days2430001--2430004, six arms per day: candidate,
frozen pressure, static7.5/static10 and cap10/cap15. Reuse V402's ORIGINAL frozen
daily-mixture probabilities. No new matching or component selection. Every day
shares 5200 exogenous sampler events; all 262 trips and 12 buses must finish.
Use scheduler high/manual on node001--006, at most32 concurrent traffic paths.

Primary development screen: candidate mean public dispatch delay is lower AND
mean actual CV2 is non-higher than EACH frozen simple mixture. Four-day bootstrap
intervals are descriptive, not independent confirmation. Report all day effects,
passenger waiting/onboard/journey, readiness, holding, admissions, rejection
reasons, and old-pressure tradeoffs. Do not require dominance of old pressure.
No confirmation, training expansion or promotion follows automatically. If the
candidate still only trades CV2 for turnover, close this candidate without a
retention sweep or predictor repair branch.

# V402: independent budget-mechanism comparison

**Decision: the preregistered joint gate failed against both simple comparators.**
Pressure-driven credit reduces actual dispatch delay, but worsens actual arrival
CV2. This establishes a turnover--regularity tradeoff, not a better tradeoff
frontier or a learned/frequency-specific advantage. The main policy is unchanged.

## Frozen comparison

All arms share manual joint, K4, the model-based fast controller, the original
12 vehicles and seven actions. Only holding constraints differ. The timetable
algorithm is fixed; realized plans can differ through vehicle supply forecasts.

The four old calibration days matched mean executed holding at 32,806.25 s/day.
Static remaining-trip credit uses c=7.5 with probability .466709 and c=10 with
probability .533291. The stop cap uses 10 s with probability .210341 and 15 s with
probability .789659. These are ex-ante daily settings; each whole-day component
was evaluated and its outcomes averaged with frozen calibration probabilities.
There is no test-day matching or within-trajectory action mixing.

Sixteen new simulation demand days, 2420001--2420016, evaluated all five physical
arms (80 paths). Each day passed the 5,200-event pairing and physical-account
checks. Engineering reproduced the old pressure policy and cap45 noop exactly.
All 147 charged paths completed without failure or replacement; zero training.

## Independent Effects

Pressure minus the frozen simple comparator; negative means lower. Intervals are
the sealed 10,000-draw paired demand-day bootstrap, not training-seed intervals.
Their upper endpoints implement the frozen 97.5% one-sided decision bounds.

| Endpoint | Static credit: mean [95% interval] | Stop cap: mean [95% interval] |
| --- | --- | --- |
| Public dispatch delay, s/day | -3744.15 [-4401.41, -3025.06] | -3123.80 [-3791.96, -2377.11] |
| Actual arrival CV2 | +.006573 [.004168, .008979] | +.006883 [.004661, .009121] |
| Journey, s/passenger | -4.761 [-6.670, -2.884] | -.056 [-1.835, 1.849] |
| Waiting, s/passenger | -.214 [-1.561, 1.093] | +.777 [-.564, 2.147] |
| Onboard, s/passenger | -4.547 [-5.536, -3.631] | -.833 [-1.770, .054] |
| Readiness delay, s/trip | -14.354 [-16.858, -11.610] | -11.975 [-14.511, -9.145] |
| Executed holding, s/trip | -3.259 [-5.413, -1.094] | -1.797 [-3.633, -.043] |

Dispatch delay falls by 17.50% and 15.03%, while CV2 rises by 10.35% and 10.89%.
Delay improves on 16/16 and 15/16 days; CV2 worsens on 14/16 and 15/16 days.
Journey improves on 15/16 static comparisons but only 9/16 cap comparisons.
The static journey benefit is mainly onboard time, not established waiting gains.

Held-out executed holding is 121.595 s/trip under pressure, versus 124.854 for
the static mixture and 123.391 for the cap mixture. Development exposure matching
does not fix test exposure or CV2. The result therefore does not isolate a pure
allocation effect at identical holding dose or identical regularity.

## Disposition

Retain this result alongside V400 as evidence for an adjustable operational
tradeoff. Do not claim that pressure credit beats simple limits at unchanged
regularity, or that double-layer learning or frequency separation is proven.
Both failed protections remain failed; no retrospective CV2 tolerance, test-day
reweighting, promotion or additional same-family parameter search is authorized.

The decisive comparison is complete. The next paper step is to present the
budget interface and measured service--turnover--regularity tradeoff, including
these strong simple controls. A dominance claim would require a substantively
new controller and fresh evaluation, not another selection from these test days.

Evidence: review/aggregate/results.json, review/selection/results.json and
review/readout/results.json. Scheduler high/manual, node002--006 for confirmation,
maximum 32 simultaneous paths. Test path median/max: 21.26/29.35 minutes. Full
traces remain on the server; only compact JSON summaries were retrieved.

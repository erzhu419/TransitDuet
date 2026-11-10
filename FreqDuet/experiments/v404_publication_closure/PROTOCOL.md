# Publication evidence closure

The user authorized completing the review gaps in one pass on 2026-11-10.
This supplement changes no policy, parameter, predictor, fleet or legal action.
V400--V403 and their decisions remain sealed. No learner is updated.

## Questions and bounded design

The four model-based factorial arms are fixed timetable/no credit, LF timetable
only, pressure credit only, and LF timetable plus pressure credit. Fixed360 with
unchanged bilateral025 is the transport baseline. Static7.5/static10 and
cap10/cap15 retain the V402 ex-ante daily-mixture probabilities. This separates
timetable, budget and interaction effects and tests allocation against simple
restrictions without changing their intensity on new outcomes.

Four predetermined regimes use nominal demand, demand scale .8, demand scale
1.2, and a fixed +1-hour peak lookup shift. Each has four distinct new demand
seeds, listed in protocol.json. All nine arms run on each seed: 144 full days,
plus two engineering paths, a hard 146-start cap. At most 32 concurrently.
Failed traffic starts count and are not automatically retried. A failed zero-traffic preflight
is retained separately. Engineering must exactly reproduce pressure on old day
2410001 and verify a zero-override native-rule path before the supplement opens.

This is a prespecified robustness/factorial supplement with four paired days
per regime, not an independently confirmed optimal frontier or route transfer.
Report paired means and descriptive day-bootstrap 95% intervals (10,000 draws,
seed 4041110). Do not pool the 144 paths as independent observations. Every arm
on a day shares the demand tape and 5,200 action-sampler events. No post-hoc
mixture weights, CV tolerances or parameter searches are permitted.

Primary readouts remain passenger waiting plus onboard exposure, actual
nonterminal CV2, readiness and original-public positive dispatch delay.
Additional readouts are passenger p90/p95 exposure, unserved/incomplete counts,
early departure counts/seconds, station-hour exposure, executed holding and
callback latency. Early departures are shown separately, not subtracted from
late departures. Day-tail waiting uses the common fixed accounting horizon.

Archive curves use the original four calibration days and label them as such;
the independent V402 points keep their original frozen mixtures. A mechanism
illustration uses the first nominal day, never the most favorable day. Future
vehicle identities are evaluation labels, not controller inputs.

## Decisions

Before any new-demand supplement path, the secondary readout was expanded to
report pressure against EACH of the four already frozen simple components,
not just their two daily mixtures. The consumed V402 operating-point plot
motivated checking whether lower dispatch delay and CV2 can coexist with more
total holding, while reporting passenger cost. All four comparisons, all seven
endpoints and all four new regimes are retained, with descriptive paired-day
intervals and no selected winner. This changes no controller, case, weight,
primary gate or path budget; it is not a rescue of V402 confirmation.

Describe the regimes in which effects and interactions repeat, including null
or adverse effects. Do not turn a significant secondary endpoint into the
failed original primary hypothesis. A claim of improved allocation at preserved
regularity would require delay improvement and CV2 nonharm against both simple
families on independent data. Otherwise the paper remains a resource-interface
and operating-tradeoff study, with that boundary stated plainly.

## Delivery

Engineering amendment, before any fresh supplement case: t214250 finished its
old-day traffic path but the new observer attempted passenger quantiles after
native cleanup and failed. That charged start is retained. Snapshot collection
now occurs at the native measurement clock, before cleanup; the original score
return is unchanged. One manually repaired pressure reproduction uses the other
engineering slot. The native portable-runtime check reuses the preregistered
nominal/2440001/fixed_bilateral cell, which is counted once among the 144 formal
cells. Thus the full cap stays 146, with no new day, policy or parameter. Native
acceptance checks only execution and zero overrides, not favorable outcomes.
Timestamp passenger durations and one-second interval integrals are retained as
separate readouts rather than being forced numerically equal. Frozen primary
endpoints keep the original interval accounting.

Run simulation/statistics on node001--006 through scheduler high and explicit
manual dispatch. Keep raw traces and frozen weights on the server. Return only
compact JSON, figures, logs and source code. Assemble a relocatable source/input
bundle server-side and test its native imports; a successful manuscript rebuild
is not traffic reproduction. Original TransitDuet and historical results remain
untouched. Release permissions and author identity are not invented.

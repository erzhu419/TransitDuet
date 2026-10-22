# V211: same-authority resource and joint-action opportunity

Freeze the script's protocol before smoke outcomes. Four consumed V106 prefixes:
actors12037/933083 x days2061001/2061004,first fresh native upper decision after7200s.
Baseline is frozen bilateral025/B200 upper. Native12-vehicle fixed pool,seven
holding bins,one shared-direction upper[-60,60] antisymmetric coefficient,EMA and900s zero-sum plan
projection unchanged. No added terminal hold,extra vehicles or fabricated slack.

The public resource ledger records ready vehicle IDs/directions,published plans,
and executed holding. Guaranteed enroute slack remains unknown;zero new credit.
Actual holding+other trip elapsed=return-launch,and commands must reproduce actual
holding. Inclusive launch delay uses positive actual-minus-original-public-clock
lateness,so delaying eligibility cannot conceal control waiting.
Truncated trips retain actual executed time and unexecuted commands separately;
incomplete service is a measured constraint outcome,not a discarded branch.

15 fixed candidates:one current-direction upper perturbation0/-30/+30s crossed
with five two-decision one-bin offset patterns. Second decision is reached causally on
another vehicle at the same station/direction. Resume bilateral025 afterward.
16 selection futures/context(960 continuations);freeze joint/upper-only/lower-only
choices before16 independent evaluation futures. Evaluate only frozen choices and
baseline,deduplicated,plus native/preserve SUFFIX controls(at most384 continuations).
Full-day costs are measured;no learned short-window value model or training.

Smoke must pass tests,cached V106 bilateral endpoints,exact no-op actions/timing,
paired-future repeat and actual two-level interventions before any formal group.
Primary:joint vs bilateral and vs selection-best single. Require passenger upper
95% bound<0 and actual CV2/readiness/inclusive delay/completion upper<=0 under BOTH
crossed actor/day and conditional-future bootstrap;both8-future evaluation halves
must have nonharmful means. No post-hoc extension,threshold change or main switch.

Limitations:four reused prefixes,two days,not new-day/general-policy confirmation.
Scenario forks preserve hidden current-window/day state:conditional simulation
lookahead is not an observable-state online planner. Suffix controls are not
whole-day B200/preserve baselines. No frequency or strong-fixed superiority claim.
All tests/replays/aggregation via scheduler high/manual on001--006;small JSON only.

Initial smoke t211032 stopped before simulation:the audit incorrectly required
two independent upper coefficients. B200 has shared_directions=true and one
coefficient. Correct the audit/intervention to the archived authority;do not
expand it. Candidate counts,futures and statistical gates stay unchanged.

Corrected source c45b2d0a93:smoke t211034 exited0 on node004;8 tests and5
rollouts passed cached endpoints,no-op actions/timing,future repeat and ledger
checks. High/manual dependent tasks:t211037 references+960 selection paths,
t211038 freeze choices,t211039 independent evaluation,t211040 aggregation.
Each waits for its predecessor's actual small marker;no evaluation before choices.
Only smoke/shard markers,choice summaries and final status sync locally.

Recovery:selection/freeze finished;independent evaluation t211039 exited1 with
63/64 groups saved. Only c1/f25 is missing:its baseline completed simulation but
the audit rejected a missing second lower opportunity before saving outcomes.
Record this as execution-support failure,not zero effect or discarded future.
No selector,action,future,metric or statistical threshold changes. Test a cached
passive replay for exact equivalence,then resume only the missing group;retain
all63 completed groups. Original aggregate stays frozen;an additional small audit
reports joint-vs-each-single,selection-to-evaluation changes and execution support.

## Outcome

Recovery source9733c89552/task t211058 exited0:9 tests,cached metrics/registered
commands exact,and only the missing four-path group replayed. Inline recovery
t211056 failed before startup(SSH argument length);the short script entry fixed
this. Original aggregate t211040 and read-only audit t211057 both exited0.
All960 selection and240 unique evaluation paths are present. Two result JSONs
(256527 bytes total) and a small marker were retrieved;no checkpoints or traces.

Joint minus comparator;negative is better. Journey is seconds/passenger;readiness
and inclusive public-clock launch delay are mean seconds/launched trip,not day sums.

| Comparator | Journey s | Actual CV2 | Readiness s | Inclusive delay s |
|---|---:|---:|---:|---:|
| bilateral025 | -.085524 | -.000022943 | -.063633 | -.051944 |
| upper-only | -.081959 | -.000049957 | -.063931 | -.065601 |
| lower-only | -.051635 | +.000011409 | -.020933 | -.008409 |
| selection-best single | -.048069 | -.000015605 | -.021231 | -.022066 |

Both primary gates fail under crossed actor/day and conditional-future intervals,
and both evaluation-half nonharm gates fail. Versus bilateral,conditional95% CI:
journey[-.225731,+.008654]s,CV2[-.000113314,+.000052967],readiness[-.232531,+.061488]s,
inclusive delay[-.221493,+.073235]s. All primary methods complete all trips.
Selection journey mean-.010291s becomes-.085524s in evaluation:the mean direction
persists,but not robust joint benefit. Only c2 selects an actual joint candidate;
it improves journey in5/16 futures(mean/SE=-.956),while mean CV2 versus bilateral
worsens. c1 is lower-only and has just1/16 nonzero endpoint effects;c0/c3 keep baseline.
The later8 futures versus bilateral have CV2+.000041508,readiness+.040196s and
inclusive delay+.050215s,with only-.001493s journey change.

One future(c1/f25),two branches lack the second eligible decision. First command
is0->0;101 later same-station/different-vehicle calls lack the required follower
support. Service clears. Preserve their true outcomes and mark execution support
failed;neither impute effects nor count these as two independent future failures.
Decision:do not promote or extend V211. Next inspect actual nonzero bin changes,
upper request-to-dispatch realization and downstream holding differences in the
consumed selection archive before designing a new persistent coordination policy.
No training,main change,new confirmation futures or threshold adjustment.

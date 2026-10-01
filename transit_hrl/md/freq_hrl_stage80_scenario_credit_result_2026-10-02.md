# Stage80 Result and Next Step

Preflight t120425/426 and full t120458-466 all exited0. Code d1dbcf686b;11 focused
tests passed.8 roots,1024 credit plus5120 independent evaluation episodes,
7,372,800 native steps;512 scenario pairs matched exactly. Source/Adam and decoder
alpha stayed frozen. Root compute174-178s, peakRAM4.5-4.7GB; no trace/checkpoint writes.

Same-scenario positive directions improve source reward, Bonferroni46 root-bootstrap CI:

| Period / Actor | Plus minus source | Plus minus minus |
| --- | --- | --- |
| 50 / upper | +.05270 [.02392,.08021] | +.10507 [.04794,.16017] |
| 50 / lower | +.56957 [.33330,.71158] | +1.17405 [.69275,1.45708] |
| 100 / upper | +.09776 [.00921,.16916] | +.19414 [.01867,.33352] |
| 100 / lower | +.99216 [.58441,1.29331] | +2.04443 [1.22198,2.66433] |

All four scenario-group gradient variance reductions are supported: geometric
rate/scenario ratios186.1,180.4,67.3,64.0 respectively. Rate positive directions
remain inconclusive versus source. Lower50 scenario-plus beats zero residual
by+.45432 CI [.17782,.68598] and rate-plus by+.81361 CI [.34032,1.36698].

Limitations: gains are small;100 upper-plus still trails zero by-.58584 CI
[-1.31307,-.04427],100 lower-plus versus zero remains inconclusive. All four cosine
improvement CIs cross zero. Mean-learning versus exploration-variance suppression
is unresolved; isolated single directions are not joint HRL evidence. Stage67 HOLD stays.

Next: freeze decoder/radius and split fresh scenario-credit directions into actor
mean-only versus log-std-only, with full-direction controls and independent evaluation.
Resolve that attribution before joint updates or long training. [Compact evidence](../results/pointmaze_scenario_credit_stage80_full_20261002_r1/qualification_compact.json).

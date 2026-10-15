# V42/V43: categorical holding and cumulative forecast bias

## Frozen protocol
Seven bins: 0/5/10/15/20/30/45 s. Use the same 12 V41 features and at most
4096 strictly earlier decisions; Torch LBFGS, non-intercept ridge 10, and
0.5 pseudo-count/bin at the prefix mean feature vector. No class balancing.
Below 32 labels, use the smoothed empirical prefix distribution.
At 2/6/10 h, propagate 64 discrete paths for 15/30 min using private RNG seeds
20261015 + draw index. Keep historical-stop-mean and rounded-response controls.
Report cumulative holding, passenger exposure/backlog, per-day errors, Monte
Carlo SE and nominal 90% bands; retain censored returns. Teacher-forced future
states/times are diagnostic labels only, never forecast inputs.

Actors 12011/933013 x demand 45007/934011 (development) or 45013/934029
(unused by V41, but used in older diagnostics). V43 reuses all eight days as
regressions, not independent confirmation. Every replay must match V35 U200/L200.
Pre-run 30-min screen, both cohorts: holding MAE <= historical, absolute bias
<=5% of actual mean, waiting/onboard MAE <=105% of historical, holding coverage
>=80%. No model/gate selection between cohorts and no main-policy switch.
Capacity, masks, passenger mass and physical reuse remain intact. All work runs
through scheduler on node001-node006; raw trajectories/checkpoints stay remote.

## Results
V42 `6316c487e4`: 22 tests and full-day smoke `t155516` pass. Eight exact replays
`t155517-t155520`; aggregate `t155521` / `45e41d9340`. At 30 min, holding MAE
833/768 s versus rounded response 675/572 and historical 337/333. Bias -494/-702 s
versus teacher-forced +76/-31 s; holding coverage 0/25%. Both screens fail.
Initial smokes `t155507/t155509` failed before rollouts (missing source dependency,
then a smoothing-frequency regression); their retries were cancelled.

V43 fixes a genuine state mismatch: reuse the existing nearest-physical-follower
ETA with journey-average speed, observable launch time, interpolated progress
and current service dwell, replacing cruise speed plus historical stop dwell.
Model/draws/gates unchanged. Source `5856651914`: 25 tests and exact full-day
smoke `t155522`; eight exact replays `t155523-t155526`, node001/004/005/006.
Aggregate `t155527` / `1c54ad854c`: holding MAE 783/690 s, bias -381/-581 s,
coverage 0/16.7%. Historical forecasts remain 337/333 s. Passenger-error screen
passes, but holding screens still fail; finite return bands cover only 19/24%.
Max fit gradient 0.000242, holding Monte Carlo SE about 12 s: solver/sampling
noise does not explain hundreds of seconds of accumulated error.

## Next Step And Limitations
Keep both negative results and the unchanged controller. Next locate remaining
headway/service-timing propagation drift before calibrating cumulative residuals
from strictly completed forecast windows, then test untouched contexts. Holding
sampling alone omits flow/travel and future-upper-replanning uncertainty; current
bands and closed-loop action values are not reliable enough for policy selection.

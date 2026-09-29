# MuJoCo v14.2 Physical-Cost Action-Router Screen

- Status: `no_behavior_safe_candidate`
- Selected arm: `None`
- Evidence role: development only; not confirmatory.
- Gate granularity: environment by disturbance mode.

| arm | conditions | trained | dual | strict | select | min slack |
|---|---:|---:|---:|---:|---:|---:|
| crossed_direct_reward | 0/15 | True | True | False | False | -1 |
| crossed_router_a004_reward | 0/15 | True | True | True | False | -25.5804 |
| crossed_router_a010_reward | 0/15 | True | True | True | False | -25.7754 |
| crossed_direct_pd_u2_l8 | 2/15 | True | False | True | False | -20.1351 |
| crossed_router_a004_pd_u2_l8 | 1/15 | False | False | True | False | -36.2009 |
| crossed_router_a010_pd_u0p5_l2 | 0/15 | False | False | True | False | -39.6 |
| crossed_router_a010_pd_u2_l8 | 0/15 | False | False | True | False | -40.4853 |
| crossed_router_a010_pd_u8_l32 | 0/15 | False | False | True | False | -40.7405 |

# MuJoCo v14.6 Conservative-Transfer Screen

- Status: `no_behavior_safe_candidate`
- Selected arm: `None`
- Evidence role: development only; not confirmatory.
- Gate granularity: environment by disturbance mode.

| arm | conditions | trained | exact params | strict | select | lower KL | lower param RMS | min slack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_s0025 | 0/15 | True | True | False | False | 0 | 0.0108078 | -1 |
| conservative_s0050 | 10/15 | True | True | True | False | 0 | 0.0108078 | -0.121317 |
| conservative_s0075 | 10/15 | True | True | True | False | 0 | 0.0108078 | -0.122373 |
| conservative_s0100 | 10/15 | True | True | True | False | 0 | 0.0108078 | -0.121066 |
| conservative_s0125 | 10/15 | True | True | True | False | 0 | 0.0108078 | -0.123688 |
| conservative_s0150 | 10/15 | True | True | True | False | 0 | 0.0108078 | -0.122114 |
| conservative_s0200 | 10/15 | True | True | True | False | 0 | 0.0108078 | -0.125491 |

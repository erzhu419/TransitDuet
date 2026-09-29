# MuJoCo v14.12 Groupwise-Robust Preflight

- Status: `do_not_expand`
- Calibration pass: `True`
- Selected arm: `None`
- Evidence role: single optimizer seed, mechanism preflight, no CI.

| arm | trained | actor | action | projection | conditions | reward signal | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled_s050_asym_u003_l008_s310_r10_k8_a000 | False | True | True | True | False | True | False |
| group_s050_asym_u003_l008_s310_r10_k8_a000 | False | True | True | False | False | True | False |
| group_s050_asym_u003_l008_s310_r05_k8_a000 | False | False | False | False | False | False | False |
| group_s050_asym_u003_l008_s310_r05_k16_a000 | False | False | False | False | False | False | False |
| group_s050_asym_u003_l008_s310_r05_k8_a001 | False | False | False | True | False | False | False |
| group_s050_asym_u003_l008_s310_r05_k8_a005 | False | False | False | True | False | False | False |
| group_s050_asym_u003_l008_s310_r10_k8_a001 | False | False | False | False | False | False | False |

This single-optimizer-seed preflight can reject a broken mechanism or authorize a larger development screen. It cannot support a performance, robustness, or statistical significance claim.

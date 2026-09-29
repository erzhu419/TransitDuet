# MuJoCo v14.14 Closed-Loop Actor-Guard Preflight

- Status: `do_not_expand`
- Calibration pass: `True`
- Selected arm: `None`
- Evidence role: single optimizer seed, mechanism preflight, no CI.

| arm | trained | actor | action | replay | trust | closed loop | selection | projection | heldout | reward signal | authorize | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| group_replay1_trust1_outer0_eps1e3_k8_control | False | False | False | True | True | True | False | True | False | False | False | False |
| group_replay1_trust1_outer0_eps5e3_k8_control | False | False | False | True | True | True | False | True | False | False | False | False |
| group_replay0_trust0_outer1_eps1e3_bt8 | False | False | False | True | True | False | False | True | False | False | False | False |
| group_replay1_trust1_outer1_eps1e3_bt4 | False | False | False | True | True | False | False | True | False | False | True | False |
| group_replay1_trust1_outer1_eps1e3_bt8 | False | False | False | True | True | False | False | True | False | False | True | False |
| group_replay1_trust1_outer1_eps5e3_bt8 | False | False | False | True | True | False | False | True | False | False | True | False |

This single-optimizer-seed preflight can reject a broken mechanism or authorize a larger development screen. It cannot support a performance, robustness, or statistical significance claim.

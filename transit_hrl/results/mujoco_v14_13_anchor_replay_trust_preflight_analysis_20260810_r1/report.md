# MuJoCo v14.13 Anchor-Replay Trust-Region Preflight

- Status: `do_not_expand`
- Calibration pass: `True`
- Selected arm: `None`
- Evidence role: single optimizer seed, mechanism preflight, no CI.

| arm | trained | actor | action | replay | trust | projection | conditions | reward signal | authorize | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| group_replay0_trust0_eps1e8_k8 | False | False | False | True | True | False | False | False | False | False |
| group_replay1_trust0_eps1e2_k8 | False | False | False | True | True | True | False | False | False | False |
| group_replay0_trust1_eps1e2_k8 | False | False | False | True | True | True | False | False | False | False |
| group_replay1_trust1_eps1e8_k8 | False | False | False | True | True | False | False | False | False | False |
| group_replay1_trust1_eps1e3_k8 | False | False | False | True | True | True | False | False | True | False |
| group_replay1_trust1_eps5e3_k8 | False | False | False | True | True | True | False | False | True | False |
| group_replay1_trust1_eps1e2_k8 | False | False | False | True | True | True | False | False | True | False |
| group_replay1_trust1_eps2e2_k8 | False | False | False | True | True | True | False | False | False | False |
| group_replay1_trust1_eps5e3_k16 | False | False | False | True | True | True | False | False | True | False |

This single-optimizer-seed preflight can reject a broken mechanism or authorize a larger development screen. It cannot support a performance, robustness, or statistical significance claim.

# MuJoCo v14.15 Closed-Loop Restoration-Filter Preflight

- Status: `expand_to_multiseed_screen`
- Calibration pass: `True`
- Selected arm: `group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3`
- Evidence role: single optimizer seed, mechanism preflight, no CI.

| arm | restoration | updates | selected violations | merit initial/final/selected | trained | actor | action | replay | trust | guard | selection | projection | heldout | reward signal | authorize | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| group_replay1_trust1_outer1_eps1e3_bt4_strict_control | False | 5 | 0/0 | 0.0554017/0/0 | True | True | True | True | True | True | True | True | True | True | False | False |
| group_replay1_trust1_outer1_restore1_eps1e3_bt4_f2 | True | 13 | 0/0 | 0.0554017/0/0 | True | True | True | True | True | True | True | True | True | True | True | True |
| group_replay1_trust1_outer1_restore1_eps1e3_bt4_f3 | True | 13 | 0/0 | 0.0554017/0/0 | True | True | True | True | True | True | True | True | True | True | True | True |
| group_replay1_trust1_outer1_restore1_eps5e3_bt4_f2 | True | 13 | 0/0 | 0.0554017/0/0 | True | True | True | True | True | True | True | True | True | True | True | True |
| group_replay1_trust1_outer1_restore1_eps5e3_bt4_f3 | True | 5 | 0/0 | 0.0554017/0/0 | True | True | True | True | True | True | False | True | False | True | True | False |
| group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3 | True | 22 | 0/0 | 0.0554017/0/0 | True | True | True | True | True | True | True | True | True | True | True | True |

This single-optimizer-seed preflight can reject a broken mechanism or authorize a larger development screen. It cannot support a performance, robustness, or statistical significance claim.

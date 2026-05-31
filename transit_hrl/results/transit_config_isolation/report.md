# Transit Config Isolation Audit

- base: `T_freqhrl_terminal.yaml`
- passed: True

| config | passed | changed paths | unexpected | missing expected |
|---|---:|---|---|---|
| T_allfreq_terminal.yaml | True | `_name, frequency.lower_mode, frequency.upper_mode, leakage.enable` | `` | `` |
| T_swapped_terminal.yaml | True | `_name, frequency.lower_mode, frequency.upper_mode, leakage.enable` | `` | `` |
| T_nopromotion_terminal.yaml | True | `_name, frequency.promotion.enable, upper.timetable_planner.promotion_replan` | `` | `` |
| T_noleakage_terminal.yaml | True | `_name, leakage.enable` | `` | `` |

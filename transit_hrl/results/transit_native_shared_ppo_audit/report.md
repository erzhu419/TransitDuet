# Native Transit Shared-PPO Interface Audit

- status: supported_interface
- config: `/home/erzhu419/mine_code/TransitDuet/transit_hrl/freq_transitduet/configs_freqduet/T_freqhrl_native_full.yaml`
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: state=20 action=4
- lower contract: state=43 action=1
- terminal dispatch: True
- promotion replan: True

| check | value |
|---|---:|
| native_runner_instantiated | True |
| uses_shared_core | True |
| upper_action_dim_matches_native | True |
| upper_action_in_bounds | True |
| lower_action_in_bounds | True |
| native_timetable_terminal_dispatch | True |
| native_promotion_replan | True |

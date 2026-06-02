# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 4
- lower contract: 43x1
- learned promotion gate: False threshold=0.62
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.0 max_hf_to_lf=0.0 max_replans=0 max_total_replans=0
- promotion replan policy: actor wait_gain_s=0.0 max_shift_s=30.0
- lower HF wait action prior: gain_s=45.0 offset=11
- off-policy replay updates per native batch: 1
- mean wait: 16.5690
- mean headway CV: 0.5165
- mean shared-PPO score: -17.6020
- mean gate value: 0.0000
- mean wait-aware replan pressure: 0.0000
- mean wait-aware replan shift: 0.0000s
- mean learned replan base delta: 0.0000s
- mean learned replan final delta: 0.0000s
- native boarded pax: 22349.0
- native alighted pax: 22349.0
- native onboard load: avg=0.5763, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.5690 | 0.5165 | -20649.7240 | 22349 | 22349 | 0.5763 | 4970 | 66 | 0 | 5232 | 52974.8188 |

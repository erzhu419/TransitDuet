# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 4
- lower contract: 43x1
- learned promotion gate: False threshold=0.55
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.0 max_hf_to_lf=0.0 max_replans=0 max_total_replans=0
- lower HF wait action prior: gain_s=0.0 offset=11
- off-policy replay updates per native batch: 1
- mean wait: 21.7040
- mean headway CV: 0.5138
- mean shared-PPO score: -22.7316
- mean gate value: 0.0000
- native boarded pax: 15982.0
- native alighted pax: 15982.0
- native onboard load: avg=0.4006, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.7040 | 0.5138 | -7929.4840 | 15982 | 15982 | 0.4006 | 4969 | 88 | 0 | 5231 | 557079.5766 |

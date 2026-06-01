# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 3
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 17x4
- upper model action dim: 4
- lower contract: 40x1
- learned promotion gate: False threshold=0.55
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- lower HF wait action prior: gain_s=0.0 offset=11
- mean wait: 7.9863
- mean headway CV: 0.4726
- mean shared-PPO score: -8.9316
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.7930 | 0.5838 | -2599.5580 | 4970 | 88 | 0 | 5232 | 3066.6198 |
| 1 | 6.4370 | 0.4045 | -3245.7870 | 4971 | 176 | 0 | 10465 | 47.2573 |
| 2 | 13.7290 | 0.4296 | -2582.0970 | 4970 | 264 | 0 | 15697 | 11.9561 |

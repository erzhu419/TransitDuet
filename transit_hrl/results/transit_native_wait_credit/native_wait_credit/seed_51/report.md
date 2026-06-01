# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 3
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 17x4
- upper model action dim: 4
- lower contract: 40x1
- learned promotion gate: False threshold=0.55
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- lower HF wait action prior: gain_s=45.0 offset=11
- mean wait: 8.2583
- mean headway CV: 0.5670
- mean shared-PPO score: -9.3923
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.3900 | 0.5902 | -7338.4160 | 4970 | 88 | 0 | 5232 | 54978.9387 |
| 1 | 6.3240 | 0.5987 | -6859.4270 | 4971 | 176 | 0 | 10465 | 10250.3229 |
| 2 | 12.0610 | 0.5120 | -7281.4520 | 4970 | 264 | 0 | 15697 | 4775.0107 |

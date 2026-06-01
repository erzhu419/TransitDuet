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
- mean wait: 4.9967
- mean headway CV: 0.5157
- mean shared-PPO score: -6.0281
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.8080 | 0.6160 | -2632.0480 | 4970 | 88 | 0 | 5232 | 3101.4678 |
| 1 | 6.1570 | 0.4318 | -3166.1120 | 4971 | 176 | 0 | 10465 | 44.8691 |
| 2 | 5.0250 | 0.4993 | -2540.6360 | 4970 | 264 | 0 | 15697 | 17.4613 |

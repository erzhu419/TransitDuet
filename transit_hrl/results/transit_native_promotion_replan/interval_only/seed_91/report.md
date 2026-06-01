# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 4
- lower contract: 43x1
- learned promotion gate: False threshold=0.62
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- mean wait: 4.6390
- mean headway CV: 0.6119
- mean shared-PPO score: -5.8628
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.6390 | 0.6119 | -18284.1200 | 4969 | 66 | 0 | 5231 | 12427.1856 |

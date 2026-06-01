# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- gate guard: strength>=0.95 age>=1.0 min_elapsed_s=900.0 cooldown_s=900.0 preselect_action=True plan_blend=0.0
- mean wait: 19.9900
- mean headway CV: 0.6037
- mean shared-PPO score: -21.1974
- mean gate value: 0.9820

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19.9900 | 0.6037 | -20039.6070 | 4971 | 74 | 31 | 5233 | 15578.8296 |

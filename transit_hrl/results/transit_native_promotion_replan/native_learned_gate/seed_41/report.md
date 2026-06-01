# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 4.3800
- mean headway CV: 0.6075
- mean shared-PPO score: -5.5950
- mean gate value: 0.9767

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.3800 | 0.6075 | -5027.4810 | 4970 | 103 | 57 | 5232 | 4580.2634 |

# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 4.9120
- mean headway CV: 0.5596
- mean shared-PPO score: -6.0312

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.9120 | 0.5596 | -10594.3170 | 4970 | 260 | 5232 | 13904.9466 |

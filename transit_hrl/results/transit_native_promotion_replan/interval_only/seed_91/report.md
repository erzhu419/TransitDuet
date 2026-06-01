# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 4.1610
- mean headway CV: 0.4257
- mean shared-PPO score: -5.0124

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.1610 | 0.4257 | -3569.1560 | 4971 | 66 | 5233 | 5125.3440 |

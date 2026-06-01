# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 43.4450
- mean headway CV: 0.4684
- mean shared-PPO score: -44.3818

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.4450 | 0.4684 | -19669.9970 | 4971 | 66 | 5233 | 44881.0947 |

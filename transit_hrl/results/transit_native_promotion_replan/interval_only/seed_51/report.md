# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 9.8580
- mean headway CV: 0.4928
- mean shared-PPO score: -10.8436

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.8580 | 0.4928 | -8887.8020 | 4970 | 66 | 5232 | 758.4721 |

# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 9.3010
- mean headway CV: 0.5432
- mean shared-PPO score: -10.3874

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.3010 | 0.5432 | -2999.2040 | 4970 | 66 | 5232 | 12467.8479 |

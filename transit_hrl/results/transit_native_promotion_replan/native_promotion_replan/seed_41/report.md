# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 4.4570
- mean headway CV: 0.4375
- mean shared-PPO score: -5.3320

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.4570 | 0.4375 | -2657.1480 | 4969 | 254 | 5231 | 106535.3798 |

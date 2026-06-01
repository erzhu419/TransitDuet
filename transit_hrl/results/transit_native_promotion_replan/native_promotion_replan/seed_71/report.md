# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 4.6100
- mean headway CV: 0.5484
- mean shared-PPO score: -5.7068

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.6100 | 0.5484 | -2701.0250 | 4970 | 93 | 5232 | 1403.2138 |

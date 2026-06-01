# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 6.3110
- mean headway CV: 0.4214
- mean shared-PPO score: -7.1538

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.3110 | 0.4214 | -3908.6900 | 4971 | 125 | 5233 | 9205.5132 |

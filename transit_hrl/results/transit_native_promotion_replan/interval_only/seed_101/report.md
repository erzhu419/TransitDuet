# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 4.5490
- mean headway CV: 0.4948
- mean shared-PPO score: -5.5386

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.5490 | 0.4948 | -2870.4730 | 4969 | 66 | 5231 | 4208.9917 |

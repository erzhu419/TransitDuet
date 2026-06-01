# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 10.1390
- mean headway CV: 0.4388
- mean shared-PPO score: -11.0166
- mean gate value: 0.9709

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.1390 | 0.4388 | -19876.7360 | 4970 | 85 | 31 | 5232 | 60274.3946 |

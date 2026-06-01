# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 4.6050
- mean headway CV: 0.3987
- mean shared-PPO score: -5.4024
- mean gate value: 0.9785

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.6050 | 0.3987 | -2691.5020 | 4970 | 166 | 141 | 5232 | 52252.2146 |

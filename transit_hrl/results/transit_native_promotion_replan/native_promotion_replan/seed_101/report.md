# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 4
- lower contract: 43x1
- learned promotion gate: False threshold=0.62
- mean wait: 48.4480
- mean headway CV: 0.2936
- mean shared-PPO score: -49.0352
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48.4480 | 0.2936 | -20437.0320 | 4971 | 135 | 0 | 5233 | 12481.3240 |

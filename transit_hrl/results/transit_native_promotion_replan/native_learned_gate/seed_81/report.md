# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 3.9910
- mean headway CV: 0.5148
- mean shared-PPO score: -5.0206
- mean gate value: 0.9731

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.9910 | 0.5148 | -4253.4860 | 4970 | 100 | 52 | 5232 | 2860.9887 |

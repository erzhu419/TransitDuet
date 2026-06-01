# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 8.5570
- mean headway CV: 0.5116
- mean shared-PPO score: -9.5802
- mean gate value: 0.9767

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.5570 | 0.5116 | -19129.1710 | 4970 | 156 | 136 | 5232 | 41666.7112 |

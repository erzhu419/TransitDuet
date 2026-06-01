# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 4
- lower contract: 43x1
- learned promotion gate: False threshold=0.62
- mean wait: 4.6570
- mean headway CV: 0.5714
- mean shared-PPO score: -5.7998
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.6570 | 0.5714 | -6313.7370 | 4970 | 130 | 0 | 5232 | 6203.4195 |

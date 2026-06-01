# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 9.4210
- mean headway CV: 0.6109
- mean shared-PPO score: -10.6428
- mean gate value: 0.9761

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.4210 | 0.6109 | -6269.6590 | 4970 | 115 | 73 | 5232 | 479.6183 |

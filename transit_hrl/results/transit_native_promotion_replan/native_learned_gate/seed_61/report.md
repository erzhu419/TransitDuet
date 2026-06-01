# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 4.6990
- mean headway CV: 0.5957
- mean shared-PPO score: -5.8904
- mean gate value: 0.9744

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.6990 | 0.5957 | -5502.4690 | 4971 | 119 | 81 | 5233 | 28728.3524 |

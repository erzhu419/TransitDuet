# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- mean wait: 22.4920
- mean headway CV: 0.3066
- mean shared-PPO score: -23.1052
- mean gate value: 0.9722

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.4920 | 0.3066 | -11152.6230 | 4971 | 125 | 87 | 5233 | 92821.0982 |

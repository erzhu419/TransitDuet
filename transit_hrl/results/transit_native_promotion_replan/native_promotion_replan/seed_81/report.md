# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- lower contract: 43x1
- mean wait: 4.8230
- mean headway CV: 0.3653
- mean shared-PPO score: -5.5536

| ep | wait | cv | reward | lower samples | upper decisions | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.8230 | 0.3653 | -2727.6020 | 4970 | 110 | 5232 | 6935.2195 |

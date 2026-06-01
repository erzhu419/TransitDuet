# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 3
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 17x4
- upper model action dim: 4
- lower contract: 40x1
- learned promotion gate: False threshold=0.55
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- lower HF wait action prior: gain_s=45.0 offset=11
- mean wait: 5.4003
- mean headway CV: 0.4600
- mean shared-PPO score: -6.3204
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.9460 | 0.3806 | -2725.1210 | 4971 | 88 | 0 | 5233 | 4309.0892 |
| 1 | 6.0480 | 0.5274 | -2725.2320 | 4971 | 176 | 0 | 10466 | 3726.0555 |
| 2 | 6.2070 | 0.4721 | -2667.5800 | 4972 | 264 | 0 | 15700 | 124.6913 |

# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 3
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 17x4
- upper model action dim: 4
- lower contract: 40x1
- learned promotion gate: False threshold=0.55
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- lower HF wait action prior: gain_s=0.0 offset=11
- mean wait: 5.4000
- mean headway CV: 0.4600
- mean shared-PPO score: -6.3201
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.9460 | 0.3806 | -2724.7000 | 4971 | 88 | 0 | 5233 | 4306.3063 |
| 1 | 6.0480 | 0.5274 | -2711.5010 | 4971 | 176 | 0 | 10466 | 3724.3695 |
| 2 | 6.2060 | 0.4721 | -2639.6940 | 4972 | 264 | 0 | 15700 | 116.3168 |

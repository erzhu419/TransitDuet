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
- mean wait: 5.1400
- mean headway CV: 0.5382
- mean shared-PPO score: -6.2164
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.1820 | 0.5590 | -4486.1890 | 4970 | 88 | 0 | 5232 | 5139.6654 |
| 1 | 4.6830 | 0.5151 | -4185.8150 | 4971 | 176 | 0 | 10465 | 198.6329 |
| 2 | 5.5550 | 0.5405 | -2949.1970 | 4970 | 264 | 0 | 15697 | 41.3110 |

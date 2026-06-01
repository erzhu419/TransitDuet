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
- mean wait: 34.2543
- mean headway CV: 0.5824
- mean shared-PPO score: -35.4191
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.2720 | 0.5102 | -15984.0210 | 4970 | 88 | 0 | 5232 | 60725.4354 |
| 1 | 30.2850 | 0.6574 | -17641.7380 | 4971 | 176 | 0 | 10465 | 11305.0595 |
| 2 | 43.2060 | 0.5795 | -17553.2220 | 4970 | 264 | 0 | 15697 | 4528.5757 |

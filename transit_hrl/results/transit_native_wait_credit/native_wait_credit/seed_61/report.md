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
- mean wait: 4.6627
- mean headway CV: 0.4612
- mean shared-PPO score: -5.5851
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.0080 | 0.5789 | -3150.0710 | 4970 | 88 | 0 | 5232 | 4710.5556 |
| 1 | 4.9380 | 0.3880 | -3138.0500 | 4971 | 176 | 0 | 10465 | 95.5657 |
| 2 | 4.0420 | 0.4167 | -3086.0840 | 4970 | 264 | 0 | 15697 | 26.2036 |

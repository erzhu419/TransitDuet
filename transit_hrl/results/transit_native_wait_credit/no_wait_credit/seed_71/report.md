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
- mean wait: 5.7660
- mean headway CV: 0.4906
- mean shared-PPO score: -6.7472
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.0950 | 0.4924 | -2841.9550 | 4971 | 88 | 0 | 5233 | 42003.0113 |
| 1 | 6.6440 | 0.5370 | -2629.6940 | 4970 | 176 | 0 | 10465 | 3815.6321 |
| 2 | 6.5590 | 0.4424 | -2809.6700 | 4970 | 264 | 0 | 15697 | 1595.6372 |

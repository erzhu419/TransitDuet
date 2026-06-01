# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- gate guard: strength>=0.95 age>=1.0 min_elapsed_s=900.0 cooldown_s=900.0 preselect_action=True plan_blend=0.0
- mean wait: 3.8140
- mean headway CV: 0.4770
- mean shared-PPO score: -4.7680
- mean gate value: 0.9820

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.8140 | 0.4770 | -7257.4350 | 4971 | 73 | 26 | 5233 | 11152.8466 |

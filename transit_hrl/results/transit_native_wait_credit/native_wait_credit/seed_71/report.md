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
- mean wait: 4.4043
- mean headway CV: 0.4362
- mean shared-PPO score: -5.2767
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.0920 | 0.4021 | -2775.9100 | 4971 | 88 | 0 | 5233 | 42146.6355 |
| 1 | 4.1550 | 0.5464 | -2643.4770 | 4970 | 176 | 0 | 10465 | 22473.1771 |
| 2 | 4.9660 | 0.3601 | -2771.4890 | 4970 | 264 | 0 | 15697 | 2317.2595 |

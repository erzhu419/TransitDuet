# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- gate guard: strength>=0.95 age>=1.0 min_elapsed_s=900.0 cooldown_s=900.0 preselect_action=True plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.1 max_hf_to_lf=8.0 max_replans=1
- lower HF wait action prior: gain_s=45.0 offset=11
- off-policy replay updates per native batch: 3
- mean wait: 38.4400
- mean headway CV: 0.5734
- mean shared-PPO score: -39.5868
- mean gate value: 0.0000

| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38.4400 | 0.5734 | -17267.5820 | 4970 | 66 | 0 | 5232 | 3950.3505 |

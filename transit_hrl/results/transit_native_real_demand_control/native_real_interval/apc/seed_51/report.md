# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 4
- lower contract: 43x1
- learned promotion gate: False threshold=0.55
- gate guard: strength>=0.0 age>=0.0 min_elapsed_s=0.0 cooldown_s=0.0 preselect_action=False plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.0 max_hf_to_lf=0.0 max_replans=0 max_total_replans=0
- lower HF wait action prior: gain_s=0.0 offset=11
- off-policy replay updates per native batch: 1
- mean wait: 49.3470
- mean headway CV: 0.5554
- mean shared-PPO score: -50.4578
- mean gate value: 0.0000
- native boarded pax: 22846.0
- native alighted pax: 22846.0
- native onboard load: avg=0.5910, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 49.3470 | 0.5554 | -20768.8750 | 22846 | 22846 | 0.5910 | 4970 | 88 | 0 | 5232 | 7722.4983 |

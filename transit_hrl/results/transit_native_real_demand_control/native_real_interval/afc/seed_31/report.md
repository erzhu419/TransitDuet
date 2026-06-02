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
- mean wait: 22.5680
- mean headway CV: 0.6205
- mean shared-PPO score: -23.8090
- mean gate value: 0.0000
- native boarded pax: 20746.0
- native alighted pax: 20746.0
- native onboard load: avg=0.5161, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.5680 | 0.6205 | -7391.2610 | 20746 | 20746 | 0.5161 | 4970 | 88 | 0 | 5232 | 50868.9317 |

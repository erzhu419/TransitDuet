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
- mean wait: 66.8480
- mean headway CV: 0.4315
- mean shared-PPO score: -67.7110
- mean gate value: 0.0000
- native boarded pax: 23218.0
- native alighted pax: 23218.0
- native onboard load: avg=0.6035, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 66.8480 | 0.4315 | -20709.2340 | 23218 | 23218 | 0.6035 | 4970 | 88 | 0 | 5232 | 22501.0024 |

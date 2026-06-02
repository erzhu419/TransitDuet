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
- mean wait: 12.3870
- mean headway CV: 0.4517
- mean shared-PPO score: -13.2904
- mean gate value: 0.0000
- native boarded pax: 18495.0
- native alighted pax: 18495.0
- native onboard load: avg=0.4963, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.3870 | 0.4517 | -11031.0200 | 18495 | 18495 | 0.4963 | 4969 | 88 | 0 | 5231 | 2208.7456 |

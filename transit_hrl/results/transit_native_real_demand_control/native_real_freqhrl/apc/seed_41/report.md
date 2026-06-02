# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 20x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.88
- gate guard: strength>=0.55 age>=1.0 min_elapsed_s=600.0 cooldown_s=900.0 preselect_action=True plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.0 max_hf_to_lf=0.0 max_replans=2 max_total_replans=0
- lower HF wait action prior: gain_s=45.0 offset=11
- off-policy replay updates per native batch: 3
- mean wait: 12.6030
- mean headway CV: 0.4937
- mean shared-PPO score: -13.5904
- mean gate value: 0.9820
- native boarded pax: 18494.0
- native alighted pax: 18494.0
- native onboard load: avg=0.4968, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.6030 | 0.4937 | -10857.7570 | 18494 | 18494 | 0.4968 | 4969 | 90 | 4 | 5231 | 1244.1203 |

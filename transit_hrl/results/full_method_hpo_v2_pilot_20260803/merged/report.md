# Full-Method Nested-Validation HPO

- protocol: `full_method_nested_hpo_v2`
- coverage: `complete`
- equal search budget: `supported`
- mechanism activity: `supported`
- held-out test accesses: `0`

| variant | candidate | rank | robust score | utility | learning | mechanism |
|---|---|---:|---:|---:|---|---|
| freq_hrl_full_v4 | freq_lr3e4_std05_activegate | 1 | +0.00129201 | +0.00568250 | ineligible | eligible |
| freq_hrl_full_v4 | freq_lr3e4_std10_balanced | 2 | +0.00373888 | +0.00766624 | ineligible | ineligible |
| freq_hrl_full_v4 | freq_lr1e4_std15_conservative | 3 | +0.00157490 | +0.00191457 | ineligible | ineligible |
| freq_hrl_full_v4 | freq_lr3e4_std15_lowgate | 4 | +0.00117894 | +0.00434263 | ineligible | ineligible |
| freq_hrl_full_v4 | freq_lr1e4_std10_balanced | 5 | +0.00077723 | +0.00162409 | ineligible | ineligible |
| freq_hrl_full_v4 | freq_lr1e4_std05_exploratory | 6 | +0.00013121 | +0.00071656 | ineligible | ineligible |
| freq_hrl_no_promotion_v4 | freq_lr3e4_std15_lowgate | 1 | +0.00615449 | +0.00749474 | eligible | not_applicable |
| freq_hrl_no_promotion_v4 | freq_lr1e4_std15_conservative | 2 | +0.00186253 | +0.00233203 | eligible | not_applicable |
| freq_hrl_no_promotion_v4 | freq_lr3e4_std05_activegate | 3 | +0.00798387 | +0.01014057 | ineligible | not_applicable |
| freq_hrl_no_promotion_v4 | freq_lr3e4_std10_balanced | 4 | +0.00718209 | +0.00859354 | ineligible | not_applicable |
| freq_hrl_no_promotion_v4 | freq_lr1e4_std05_exploratory | 5 | +0.00204846 | +0.00262532 | ineligible | not_applicable |
| freq_hrl_no_promotion_v4 | freq_lr1e4_std10_balanced | 6 | +0.00184233 | +0.00235251 | ineligible | not_applicable |
| freq_hrl_no_hf_lower_v4 | freq_lr3e4_std05_activegate | 1 | +0.00446754 | +0.00856555 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v4 | freq_lr3e4_std15_lowgate | 2 | +0.00305696 | +0.00358309 | ineligible | not_applicable |
| freq_hrl_no_hf_lower_v4 | freq_lr1e4_std10_balanced | 3 | +0.00129962 | +0.00138793 | ineligible | not_applicable |
| freq_hrl_no_hf_lower_v4 | freq_lr1e4_std05_exploratory | 4 | +0.00118490 | +0.00147829 | ineligible | not_applicable |
| freq_hrl_no_hf_lower_v4 | freq_lr1e4_std15_conservative | 5 | -0.00021470 | +0.00110561 | ineligible | not_applicable |
| freq_hrl_no_hf_lower_v4 | freq_lr3e4_std10_balanced | 6 | -0.00024215 | +0.00327875 | ineligible | not_applicable |
| freq_hrl_no_leakage_v4 | freq_lr3e4_std10_balanced | 1 | +0.00393261 | +0.00782140 | ineligible | not_applicable |
| freq_hrl_no_leakage_v4 | freq_lr3e4_std05_activegate | 2 | +0.00174986 | +0.00598235 | ineligible | not_applicable |
| freq_hrl_no_leakage_v4 | freq_lr1e4_std15_conservative | 3 | +0.00153701 | +0.00189853 | ineligible | not_applicable |
| freq_hrl_no_leakage_v4 | freq_lr3e4_std15_lowgate | 4 | +0.00117776 | +0.00445399 | ineligible | not_applicable |
| freq_hrl_no_leakage_v4 | freq_lr1e4_std10_balanced | 5 | +0.00046264 | +0.00151181 | ineligible | not_applicable |
| freq_hrl_no_leakage_v4 | freq_lr1e4_std05_exploratory | 6 | +0.00012808 | +0.00084695 | ineligible | not_applicable |
| flat_ppo_matched_v4 | ppo_lr1e4_std05 | 1 | -0.00367739 | -0.00348207 | ineligible | not_applicable |
| flat_ppo_matched_v4 | ppo_lr1e4_std10 | 2 | -0.00368655 | -0.00351691 | ineligible | not_applicable |
| flat_ppo_matched_v4 | ppo_lr1e4_std15 | 3 | -0.00370476 | -0.00345315 | ineligible | not_applicable |
| flat_ppo_matched_v4 | ppo_lr3e4_std05 | 4 | -0.00371830 | -0.00346964 | ineligible | not_applicable |
| flat_ppo_matched_v4 | ppo_lr3e4_std10 | 5 | -0.00371830 | -0.00346964 | ineligible | not_applicable |
| flat_ppo_matched_v4 | ppo_lr3e4_std15 | 6 | -0.00371830 | -0.00346964 | ineligible | not_applicable |
| flat_gru_ppo_matched_v4 | ppo_lr3e4_std05 | 1 | -0.00031972 | -0.00012496 | ineligible | not_applicable |
| flat_gru_ppo_matched_v4 | ppo_lr1e4_std05 | 2 | -0.00068021 | -0.00060900 | ineligible | not_applicable |
| flat_gru_ppo_matched_v4 | ppo_lr1e4_std10 | 3 | -0.00072091 | -0.00066321 | ineligible | not_applicable |
| flat_gru_ppo_matched_v4 | ppo_lr1e4_std15 | 4 | -0.00084005 | -0.00067448 | ineligible | not_applicable |
| flat_gru_ppo_matched_v4 | ppo_lr3e4_std10 | 5 | -0.00130212 | -0.00053872 | ineligible | not_applicable |
| flat_gru_ppo_matched_v4 | ppo_lr3e4_std15 | 6 | -0.00150890 | -0.00097110 | ineligible | not_applicable |
| generic_hrl_ppo_matched_v4 | ppo_lr3e4_std05 | 1 | +0.00142333 | +0.00301280 | eligible | not_applicable |
| generic_hrl_ppo_matched_v4 | ppo_lr3e4_std10 | 2 | +0.00094525 | +0.00238047 | eligible | not_applicable |
| generic_hrl_ppo_matched_v4 | ppo_lr3e4_std15 | 3 | +0.00040070 | +0.00170006 | eligible | not_applicable |
| generic_hrl_ppo_matched_v4 | ppo_lr1e4_std05 | 4 | +0.00004895 | +0.00059367 | eligible | not_applicable |
| generic_hrl_ppo_matched_v4 | ppo_lr1e4_std10 | 5 | -0.00013253 | +0.00043581 | eligible | not_applicable |
| generic_hrl_ppo_matched_v4 | ppo_lr1e4_std15 | 6 | -0.00025712 | +0.00027667 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v4 | ppo_lr3e4_std05 | 1 | +0.00172817 | +0.00262024 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v4 | ppo_lr3e4_std10 | 2 | +0.00147200 | +0.00205358 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v4 | ppo_lr3e4_std15 | 3 | +0.00091068 | +0.00156976 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v4 | ppo_lr1e4_std05 | 4 | +0.00041834 | +0.00070381 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v4 | ppo_lr1e4_std10 | 5 | +0.00033286 | +0.00053944 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v4 | ppo_lr1e4_std15 | 6 | +0.00019178 | +0.00041989 | eligible | not_applicable |
| flat_sac_matched_v4 | off_lr3e4_w1024_b64 | 1 | +0.00646404 | +0.01235803 | eligible | not_applicable |
| flat_sac_matched_v4 | off_lr3e4_w2048_b64 | 2 | +0.00574519 | +0.00753344 | eligible | not_applicable |
| flat_sac_matched_v4 | off_lr1e3_w4096_b64 | 3 | +0.00376655 | +0.00726584 | eligible | not_applicable |
| flat_sac_matched_v4 | off_lr1e4_w2048_b64 | 4 | +0.00118528 | +0.00240963 | eligible | not_applicable |
| flat_sac_matched_v4 | off_lr3e4_w2048_b128 | 5 | +0.00113133 | +0.00635485 | eligible | not_applicable |
| flat_sac_matched_v4 | off_lr1e4_w1024_b64 | 6 | +0.00049632 | +0.00462072 | eligible | not_applicable |
| flat_td3_matched_v4 | off_lr3e4_w2048_b128 | 1 | +0.01794469 | +0.02265838 | eligible | not_applicable |
| flat_td3_matched_v4 | off_lr1e4_w2048_b64 | 2 | +0.01351936 | +0.02144269 | eligible | not_applicable |
| flat_td3_matched_v4 | off_lr1e3_w4096_b64 | 3 | +0.01234357 | +0.02227794 | eligible | not_applicable |
| flat_td3_matched_v4 | off_lr3e4_w2048_b64 | 4 | +0.01210410 | +0.01830324 | eligible | not_applicable |
| flat_td3_matched_v4 | off_lr3e4_w1024_b64 | 5 | +0.01137403 | +0.01990469 | eligible | not_applicable |
| flat_td3_matched_v4 | off_lr1e4_w1024_b64 | 6 | +0.01027892 | +0.01944319 | eligible | not_applicable |

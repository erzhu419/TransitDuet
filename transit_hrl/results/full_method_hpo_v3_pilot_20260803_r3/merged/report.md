# Full-Method Nested-Validation HPO

- protocol: `full_method_nested_hpo_v3`
- coverage: `complete`
- equal search budget: `supported`
- mechanism activity: `supported`
- held-out test accesses: `0`

| variant | candidate | rank | robust score | utility | learning | mechanism |
|---|---|---:|---:|---:|---|---|
| freq_hrl_full_v5 | v5_u3_l3_h1_p3_activegate | 1 | +0.01057506 | +0.01332044 | eligible | eligible |
| freq_hrl_full_v5 | v5_u5_l3_h005_p01_macro | 2 | +0.00739759 | +0.01159601 | eligible | ineligible |
| freq_hrl_full_v5 | v5_u3_l1_h3_p01_tactical | 3 | +0.00373476 | +0.00675752 | eligible | ineligible |
| freq_hrl_full_v5 | v5_u1_l3_h005_p01_tracking | 4 | +0.00078501 | +0.00114660 | eligible | ineligible |
| freq_hrl_full_v5 | v5_u3_l3_h003_p003_conservative | 5 | +0.00285350 | +0.00353211 | ineligible | ineligible |
| freq_hrl_full_v5 | v5_u3_l3_h01_p005_balanced | 6 | +0.00228025 | +0.00374196 | ineligible | ineligible |
| freq_hrl_no_promotion_v5 | v5_u3_l1_h3_p01_tactical | 1 | +0.00659626 | +0.01248313 | eligible | not_applicable |
| freq_hrl_no_promotion_v5 | v5_u3_l3_h1_p3_activegate | 2 | +0.00628789 | +0.00784486 | eligible | not_applicable |
| freq_hrl_no_promotion_v5 | v5_u3_l3_h01_p005_balanced | 3 | +0.00611400 | +0.00759263 | eligible | not_applicable |
| freq_hrl_no_promotion_v5 | v5_u5_l3_h005_p01_macro | 4 | +0.00600734 | +0.01558742 | eligible | not_applicable |
| freq_hrl_no_promotion_v5 | v5_u3_l3_h003_p003_conservative | 5 | +0.00527136 | +0.00652043 | eligible | not_applicable |
| freq_hrl_no_promotion_v5 | v5_u1_l3_h005_p01_tracking | 6 | +0.00132418 | +0.00325288 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v5 | v5_u3_l3_h1_p3_activegate | 1 | +0.01126978 | +0.01412295 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v5 | v5_u5_l3_h005_p01_macro | 2 | +0.00503647 | +0.00596038 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v5 | v5_u3_l3_h01_p005_balanced | 3 | +0.00468378 | +0.00623226 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v5 | v5_u3_l3_h003_p003_conservative | 4 | +0.00243277 | +0.00525802 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v5 | v5_u3_l1_h3_p01_tactical | 5 | +0.00059036 | +0.00701611 | eligible | not_applicable |
| freq_hrl_no_hf_lower_v5 | v5_u1_l3_h005_p01_tracking | 6 | -0.00006604 | +0.00045983 | eligible | not_applicable |
| freq_hrl_no_leakage_v5 | v5_u3_l3_h1_p3_activegate | 1 | +0.01122384 | +0.01366512 | eligible | not_applicable |
| freq_hrl_no_leakage_v5 | v5_u5_l3_h005_p01_macro | 2 | +0.00798040 | +0.01140987 | eligible | not_applicable |
| freq_hrl_no_leakage_v5 | v5_u3_l1_h3_p01_tactical | 3 | +0.00352359 | +0.00689984 | eligible | not_applicable |
| freq_hrl_no_leakage_v5 | v5_u1_l3_h005_p01_tracking | 4 | +0.00069566 | +0.00109293 | eligible | not_applicable |
| freq_hrl_no_leakage_v5 | v5_u3_l3_h003_p003_conservative | 5 | +0.00292172 | +0.00353345 | ineligible | not_applicable |
| freq_hrl_no_leakage_v5 | v5_u3_l3_h01_p005_balanced | 6 | +0.00226597 | +0.00368746 | ineligible | not_applicable |
| flat_ppo_matched_v5 | ppo_lr3e4_std05 | 1 | -0.00360615 | -0.00296277 | ineligible | not_applicable |
| flat_ppo_matched_v5 | ppo_lr1e4_std05 | 2 | -0.00371189 | -0.00350687 | ineligible | not_applicable |
| flat_ppo_matched_v5 | ppo_lr1e4_std10 | 3 | -0.00371189 | -0.00352772 | ineligible | not_applicable |
| flat_ppo_matched_v5 | ppo_lr1e4_std15 | 4 | -0.00371189 | -0.00354540 | ineligible | not_applicable |
| flat_ppo_matched_v5 | ppo_lr3e4_std10 | 5 | -0.00371189 | -0.00357247 | ineligible | not_applicable |
| flat_ppo_matched_v5 | ppo_lr3e4_std15 | 6 | -0.00377997 | -0.00369934 | ineligible | not_applicable |
| flat_gru_ppo_matched_v5 | ppo_lr1e4_std15 | 1 | -0.00083540 | -0.00051623 | ineligible | not_applicable |
| flat_gru_ppo_matched_v5 | ppo_lr3e4_std05 | 2 | -0.00087586 | +0.00015184 | ineligible | not_applicable |
| flat_gru_ppo_matched_v5 | ppo_lr3e4_std10 | 3 | -0.00087586 | -0.00037656 | ineligible | not_applicable |
| flat_gru_ppo_matched_v5 | ppo_lr1e4_std05 | 4 | -0.00090193 | -0.00047639 | ineligible | not_applicable |
| flat_gru_ppo_matched_v5 | ppo_lr1e4_std10 | 5 | -0.00091261 | -0.00048483 | ineligible | not_applicable |
| flat_gru_ppo_matched_v5 | ppo_lr3e4_std15 | 6 | -0.00102344 | -0.00078621 | ineligible | not_applicable |
| generic_hrl_ppo_matched_v5 | ppo_lr3e4_std05 | 1 | +0.00297549 | +0.00435084 | eligible | not_applicable |
| generic_hrl_ppo_matched_v5 | ppo_lr3e4_std10 | 2 | +0.00214334 | +0.00319291 | eligible | not_applicable |
| generic_hrl_ppo_matched_v5 | ppo_lr3e4_std15 | 3 | +0.00168169 | +0.00272141 | eligible | not_applicable |
| generic_hrl_ppo_matched_v5 | ppo_lr1e4_std05 | 4 | +0.00083919 | +0.00098969 | eligible | not_applicable |
| generic_hrl_ppo_matched_v5 | ppo_lr1e4_std10 | 5 | +0.00062786 | +0.00071874 | eligible | not_applicable |
| generic_hrl_ppo_matched_v5 | ppo_lr1e4_std15 | 6 | +0.00052239 | +0.00061253 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v5 | ppo_lr3e4_std05 | 1 | +0.00309252 | +0.00396081 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v5 | ppo_lr3e4_std10 | 2 | +0.00215306 | +0.00308615 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v5 | ppo_lr1e4_std05 | 3 | +0.00083220 | +0.00101777 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v5 | ppo_lr1e4_std10 | 4 | +0.00057836 | +0.00080098 | eligible | not_applicable |
| generic_hrl_gru_ppo_matched_v5 | ppo_lr3e4_std15 | 5 | +0.00143178 | +0.00281828 | ineligible | not_applicable |
| generic_hrl_gru_ppo_matched_v5 | ppo_lr1e4_std15 | 6 | +0.00039007 | +0.00069555 | ineligible | not_applicable |
| flat_sac_matched_v5 | off_lr3e4_w1024_b64 | 1 | +0.00908709 | +0.01220481 | eligible | not_applicable |
| flat_sac_matched_v5 | off_lr1e3_w4096_b64 | 2 | +0.00559307 | +0.00710771 | eligible | not_applicable |
| flat_sac_matched_v5 | off_lr3e4_w2048_b64 | 3 | +0.00476034 | +0.00794928 | eligible | not_applicable |
| flat_sac_matched_v5 | off_lr3e4_w2048_b128 | 4 | +0.00256926 | +0.00528994 | eligible | not_applicable |
| flat_sac_matched_v5 | off_lr1e4_w1024_b64 | 5 | -0.00027188 | +0.00168931 | eligible | not_applicable |
| flat_sac_matched_v5 | off_lr1e4_w2048_b64 | 6 | -0.00428716 | -0.00075980 | eligible | not_applicable |
| flat_td3_matched_v5 | off_lr3e4_w2048_b128 | 1 | +0.01836279 | +0.02263561 | eligible | not_applicable |
| flat_td3_matched_v5 | off_lr3e4_w1024_b64 | 2 | +0.01648346 | +0.02181093 | eligible | not_applicable |
| flat_td3_matched_v5 | off_lr3e4_w2048_b64 | 3 | +0.00753851 | +0.01522779 | eligible | not_applicable |
| flat_td3_matched_v5 | off_lr1e4_w2048_b64 | 4 | +0.00499978 | +0.01668615 | eligible | not_applicable |
| flat_td3_matched_v5 | off_lr1e3_w4096_b64 | 5 | +0.00377242 | +0.01801890 | eligible | not_applicable |
| flat_td3_matched_v5 | off_lr1e4_w1024_b64 | 6 | -0.00755586 | +0.00922787 | eligible | not_applicable |

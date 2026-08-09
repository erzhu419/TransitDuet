# Freq-HRL v7.3.2 Confirmatory Results

- status: `valid`
- independent training replicates: `24`
- held-out paths per replicate: `8`
- plan SHA-256: `6c72874ad3260a23da30753746f423f9b4288dfb930a14c2ae2728e92b2aa5ad`

## Primary Pooled Contrasts

- flat_ppo_matched_v7 / total_return: `supported_improvement` (directional delta 0.00503311, Holm p=0.00042)
- flat_ppo_matched_v7 / LowerLFDriftAbs: `supported_improvement` (directional delta 0.00477481, Holm p=6e-05)
- flat_gru_ppo_matched_v7 / total_return: `supported_improvement` (directional delta 0.00421103, Holm p=0.01506)
- flat_gru_ppo_matched_v7 / LowerLFDriftAbs: `supported_improvement` (directional delta 0.00392772, Holm p=6e-05)
- generic_hrl_ppo_matched_v7 / total_return: `supported_improvement` (directional delta 0.00344891, Holm p=0.00028)
- generic_hrl_ppo_matched_v7 / LowerLFDriftAbs: `inconclusive` (directional delta -5.64634e-06, Holm p=0.6546)
- generic_hrl_gru_ppo_matched_v7 / total_return: `supported_harm` (directional delta -0.00240725, Holm p=0.01506)
- generic_hrl_gru_ppo_matched_v7 / LowerLFDriftAbs: `inconclusive` (directional delta 3.26793e-05, Holm p=0.6546)
- flat_sac_matched_v7 / total_return: `inconclusive` (directional delta 0.000966529, Holm p=0.6546)
- flat_sac_matched_v7 / LowerLFDriftAbs: `supported_improvement` (directional delta 0.00725949, Holm p=6e-05)
- flat_td3_matched_v7 / total_return: `supported_improvement` (directional delta 0.00271026, Holm p=0.01674)
- flat_td3_matched_v7 / LowerLFDriftAbs: `supported_improvement` (directional delta 0.0127159, Holm p=6e-05)

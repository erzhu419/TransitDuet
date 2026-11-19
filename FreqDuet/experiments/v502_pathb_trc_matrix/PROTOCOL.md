# V502 Gate 3 Protocol — Full TR-C Confirmation Matrix, Literature Baselines, 4-Way Frequency Ablation & OOD Suite

## 1. Strictly Untouched Confirmation Split
- **Primary Confirmation Days (`16` paired CRN days per corridor)**: `2700101..2700116` on both `tidal_bimodal` and `mbta_route111_wkdy` (strictly disjoint from Gate 1 dev days `2600101..2600108`, RL training episodes `1000000+`, and Gate 2 validation days `2650101..2650110`).
- **Out-of-Distribution (OOD) & Null-Regime Generalization (`8` paired CRN days `2710101..2710108` per regime, zero retraining)**:
  1. `ood_peak_shift` (`peak_shift_override = -1` on half, `+1` on half): unannounced $\pm 1\text{ h}$ rush-hour shift;
  2. `ood_hf_surge` (`burst_scale = 1.8`): severe localized station surge shock;
  3. `ood_congestion` (`speed_delta_mps = -0.8`): corridor-wide traffic slowdown tightening turnaround slack;
  4. `null_flat_sanity`: stationary demand corridor (`flat_sanity`) verifying zero-harm behavior.

## 2. Compared Methods (12 Arms Total)
### A. Classical & Optimization Baselines
1. `B0_Fixed_NoHold`: Constant 360 s timetable, zero stop holding.
2. `B1_Daganzo2009`: Adaptive forward-headway holding (Daganzo, *TR-B* 2009) under fixed timetable.
3. `B2_Berrebi2018`: Causal AVL bilateral prediction holding (Berrebi et al., *TR-C* 2018).
4. `B3_EconMPC`: Rolling-horizon schedule + turnaround-slack + Osuna-Newell holding Economic MPC (Delgado et al., *TR-C* 2012; Gkiotsalitis & Cats, *TR-C* 2021).

### B. Deep RL Baselines (`200` Episodes × Independent Seeds)
5. `B4_FlatSAC`: Monolithic single-level stop-event Constrained Discrete SAC without upper slow-clock timetable/budget hierarchy (Wang & Sun, *TR-C* 2020).
6. `B5_TransitDuet_NoFreq`: Matched Bi-Level Constrained SAC without frequency decomposition (`band_mode="nofreq"`).

### C. FreqDuet & Matched 4-Way Frequency / Mechanism Ablations (`200` Episodes × Independent Seeds)
7. `FreqDuet_Aligned` (**Proposed**): Pure Bi-Level Constrained SAC with `Upper-LF + Lower-HF`, persistent residual promotion, and `TurnaroundBudgetLedger` hard masking.
8. `Abl_RawHistory`: Matched B1 with unfiltered causal APC histories (`band_mode="raw"`).
9. `Abl_AllBand`: Matched B1 feeding both LF and HF unseparated to both levels (`band_mode="all_band"`).
10. `Abl_Swapped`: Matched B1 with inverted band assignment `Upper-HF + Lower-LF` (`band_mode="swapped"`).
11. `Abl_NoBudget`: `FreqDuet_Aligned` without turnaround holding budget (`ablation="no_budget"`).
12. `Abl_NoPromotion`: `FreqDuet_Aligned` without persistent residual-to-LF promotion (`ablation="no_promotion"`).

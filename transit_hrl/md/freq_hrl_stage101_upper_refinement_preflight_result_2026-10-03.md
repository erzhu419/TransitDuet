# Stage101 Upper Refinement: Preflight Result

`t129038/t129039` both finished with exit0 on node006. Official qualification exactly matches read-only reaggregation: `preflight_passed`, mechanical gate passed. Both upper means start from fixed Stage98 UJ; Stage99 lowers, std, values, source Adam, forecaster and decoder remain frozen.

Native cost: 152 episodes / 45,600 steps, eight upper updates, six donor loads, four UJ initialization checks, no checkpoint or upper-replay writes. Native wall time: 42.56 s. Only 17,194 bytes of compact JSON were retrieved; weights and raw trajectories remain on the server.

Short refined-minus-fixed-UJ diagnostics: at50 common +0.002156, independent +0.002301; at100 common +0.000826, independent +0.000514. One root, H300, two updates and four evaluation paths per composition do not establish performance improvement. Matched-upper specialization remains a separate diagnostic.

Next: submit the unchanged preregistered eight-root H1200 full cohort, with eight updates and 32 final evaluation paths per composition. Require all four corrected same-lower refined-minus-fixed-UJ CI lower bounds to be positive. No parameter, donor or root selection from these short results; Stage100 negative contrasts and Stage67 HOLD remain recorded.

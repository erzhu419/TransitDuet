# Stage242: fresh-future prefix and refit transfer

Run `native_transit_fresh_transfer_stage242_development_20261015_r1`: all34 tasks
done/0 (prepare t155550, evaluations t155558-t155589, aggregate t155597).
Seven tests passed. All470,400 old metric records and80,640 sealed actions reproduce
exactly; pooled streaming aggregation differs from Stage240 by <=1.12e-16.

## Design

Reuse all240 states/40 prefixes, eight Stage241 futures and all five cached Stage240
endpoints. Compare seen54, OOF probe6, train60 and held60 with old inner2/outer4/matched
labels and fresh4+4/8. Actions are fixed before reading fresh labels. No new fits or
episodes. Only the7,471,791-byte reduced aggregate was pulled; full metrics stay remote.

## Results

Mixed pooled joint/raw/half-safe on held60/fresh8; negative is better, wait in minutes.

| Endpoint | Cost vs own raw | Wait vs own raw | Cost vs full raw |
|---|---:|---:|---:|
| Cached54/2 | +0.000030333 | +0.000019551 | +0.000917082 |
| Residual60/2, fixed54 reference | -0.000006564 | -0.000013079 | +0.000880186 |
| Native60/2 | -0.000107727 | -0.000279883 | +0.001090472 |
| Residual60/4, fixed60/2 reference | -0.000112577 | -0.000286784 | +0.001085622 |
| Native60/4 | +0.000047177 | +0.000101172 | +0.000047177 |

Native60/4 improves train-prefix fresh cost/wait (-0.000124997/-0.000147005), but
worsens both on held prefixes. All-MSE/original/pooled also loses both after native
full refit on held prefixes (+0.000062695/+0.000188151); its Stage241 signal is not a repair.
For this all-MSE secondary, cached54/2 seen-state MSE improves on old fit labels
(0.00351556 ->0.00344476), but worsens on fresh labels (0.00154324 ->0.00159577).
Native60/4 shows the same old-fit versus fresh-seen reversal. These are concurrent
noise-fitting and prefix/reference-transfer symptoms, not a capacity diagnosis or seed-CI proof.
Fleet explains +0.000024414 of cached54/2 held cost harm, but full-native harm also
includes +0.000025476 nonfleet cost. A fleet-only gate would not address both failures.

## Next Step And Limits

Prioritize variance-aware, group-regularized residual estimation with separate
service/risk terms and held-prefix validation, rather than encoder expansion or another gate.
Reused development data and repeated balanced blocks, not fresh seed CI or complete
end-to-end HRL. The all-MSE secondary was named after Stage241 inspection. Defaults
and reserved Stage169 cohorts remain unchanged. The MC adjunct finds Stage241 joint
cost is only0.18 sampling SE from zero; rare fleet events dominate cost uncertainty.

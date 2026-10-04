# Stage109 Current-Lower Response Result

Preflight t134578/t134579 and full t134580-t134588 are all done/exit0. Full used all8 final Stage107 learned-hint lowers,both periods,8 fresh forecast-only trajectories/period. All392 physical/history feedback features remained identical across counterfactual advice. All128 forecast/zero-action/feedback/network checks passed;5 focused tests passed. Independent server-side reaggregation of316 scalar summaries differs from the official output by at most2.17e-19.

Equal-root means;command denotes tanh of the lower Gaussian mean,compared with forecast on the same states:

| Period | Legacy learned-mean command RMS | Full-scale learned-mean command RMS | Full-scale sampled-plan command RMS | Full-scale sampled-plan mean KL |
| --- | ---: | ---: | ---: | ---: |
| 50 | 5.812e-7 | 2.558e-5 | 2.741e-4 | 2.986e-6 |
| 100 | 3.974e-7 | 5.290e-6 | 8.644e-5 | 2.565e-7 |

The legacy alpha0.0214-0.0971 materially attenuates the current optional-advice channel. Full-scale central-secant Jacobian Frobenius RMS is44.04x/13.59x the legacy value. Both equal-root mean singular values are nonzero. Nevertheless,the learned upper mean is weak:latent mean RMS0.03234/0.02681 versus fixed std0.40155. Full-scale learned-sample minus paired zero-mean-noise command RMS is only2.359e-5/4.790e-6. Increasing alpha therefore amplifies predominantly sampled residuals,not an established useful learned plan.

Cost:128 native episodes/153,600 steps;30,720 common states,768,000 hypothetical lower-mean rows,2,304 upper-distribution rows and52,992 residual-curve decodes. No policy updates,critic fits,new checkpoint or raw-trace writes. Native/probe wall11.25-12.13s/root after source loading;inherited preparation remains separate. Scheduler assigned node001/004/005/006 dynamically. RAM peaks were not sampled;recorded zeros are not zero memory usage. [Compact result](../results/pointmaze_control_response_stage109_full_20261004_r1/compact_summary.json),[independent aggregation](../results/pointmaze_control_response_stage109_full_20261004_r1/aggregation_reproduction.json),[terminal roster](../results/pointmaze_control_response_stage109_full_20261004_r1/scheduler_tasks.json).

## Next

Do not launch an alpha sweep or extend the old donor reward matrix. Next measure action-conditioned native closed-loop reward differences with the current fixed lower and existing full-scale decoder,using fresh paired interventions and forecast controls. Establish that command directions have useful task gain before training a current-lower upper policy with paired action-conditioned credit. If the response produces no useful task gain,the next repair belongs in the trainable plan/lower interface,not more seeds or gate tuning.

## Limitations

These are conditional same-state policy responses,not actual closed-loop performance gains. .001 is only the earlier lower-update KL reference;mean KL below it is not a safety/no-harm bound. Quantiles in the compact summary are averaged episode quantiles,not a pooled percentile. This diagnostic does not select a production alpha,prove the sole cause of Stage108's negative result,or change its hierarchy claim boundary. Stage67 critic HOLD and all prior negative evidence remain in place.

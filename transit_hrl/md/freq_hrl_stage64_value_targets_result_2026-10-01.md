# Stage64 Value Targets Result

Four tests t116702/node005 passed (45.286s). Actual archive preflight t116706/node005 and qualification t116709/node005 completed with exit0. All four raw GAE controls exactly reproduce Stage63's pre-actor episode-MC probe; actor/upper networks and Adam remain frozen.

Preflight EV ranges:gae_raw0.01041-0.01096,mc_raw0.01025-0.01091,gae_normalized0.06457-0.07712,mc_normalized0.06329-0.07342. MC without normalization barely changes fitting; both normalized treatments improve early variance fitting. The preflight uses only2 calibration iterations and does not pass the fixed0.10 candidate EV floor. This is a directional diagnostic, not grounds to lower the threshold or choose another candidate.

Cost:24 archived episodes,7200 lower/108 upper reconstructions plus21600 extra critic scalar calls;32 critic calibration calls,128 matched value optimizer/forward steps (32 per treatment,64 MC-supervised steps),4800 initialization prediction rows and32 representation batches. All16 critic checkpoints remain remote. No actor optimizer steps, new native paths or forecaster fitting; only15KB compact JSON pulled.

## Next

Run the frozen eight roots,16 calibration iterations/eight paths. Preserve every period/arm, including root310037. Keep mc_normalized as the preregistered candidate and require every full case EV>=0.10 with MSE below gae_raw. A pass is critic-only and still needs a separate guarded actor/native trial; Stage63 native evaluation remains HOLD.

Full preregistration committed as662c7b4032 before outcomes. All eight tasks t116713-t116720 have started dynamically on node004/005/006, with9CPU each and no node pin. Early measured RAM is approximately3.9GB/task; submitted RAM is12GB. Full qualification is pending. No raw archives or critic weights are pulled locally.

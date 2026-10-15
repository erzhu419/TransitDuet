# V55: completed-prequential road/time residual bias

Freeze V54's persistence and the real controller. Form actual-minus-predicted
leg residuals using each earlier origin's stored V54 model, never a current
in-sample refit. Fit smooth short/long horizon offsets with global, road and
road/time partial pooling (the existing ten pseudo-pairs). Only end_s < now
labels in the three-hour / 4096-pair prefix enter a fit; simultaneous and
unfinished legs do not. This calibration is additive to V54, not recursively
trained on its own corrected forecasts.

Keep the existing ratio noise, holding and passenger processes. Positive
duration truncation uses the existing dt convention and is counted in scores.
Smoke must preserve V54 draws exactly for zero correction. The unchanged gate
requires each cohort's short-lead MAE <= V50 and long-lead MAE <= V48 at the six
V52 clocks. Only a passing prefix model proceeds to a 72-origin closed-loop
screen; no main-policy switch. All six demand seeds remain development data.

## Results
Smoke t160222 passes 21 tests; preparation t162143-t162147 covers all 12 days.
Gate t162148 fails in every cohort. Short-lead MAE delta versus V50 +2.190 s
[1.755,2.678], versus V54 +3.535 s [3.282,3.791]; long-lead delta versus V48
+6.604 s [6.122,7.067], versus V54 +5.441 s [5.081,5.812]. There are 17 physical
floor predictions. No closed-loop matrix or policy promotion is launched.

Alignment audit t162150 completed. Correction/error correlation is -0.196
[-0.220,-0.176] for 0-15 min and -0.242 [-0.269,-0.219] for 15-30 min.
Mean correction/error products are -81.141 and -153.006 s^2, while correction
energies are 147.741 and 223.515 s^2. MSE increases by 310.023 [283.184,343.075]
and 529.527 [491.432,573.570] s^2. These six-demand-cluster results show wrong
aggregate alignment, not just excessive amplitude: a uniform positive shrink
factor cannot improve aggregate MSE when its correction/error product is
negative. No retrospective attenuation or sign flip is fitted or deployed.
Close V55 promotion and retain V54 as the stronger forecast candidate, not a
passed long-horizon model. The alignment audit explains this negative result;
it does not authorize retrospective attenuation or another bias patch. The
next experiment is V56 benefit timing on the original V37/V38 action bank.

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

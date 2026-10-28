#!/usr/bin/gnuplot
#  _  __          _      __  __          _   _
# | |/ /    /\   | |    |  \/  |   /\   | \ | |
# | ' /    /  \  | |    | \  / |  /  \  |  \| |
# |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
# | . \  / ____ \| |____| |  | |/ ____ \| |\  |
# |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

# Kalman Filter
# Version 0.5.4
# https://github.com/FrancoisCarouge/Kalman

# SPDX-License-Identifier: Unlicense

# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <https://unlicense.org>

set terminal svg enhanced background rgb "white" size 720,1440
set datafile separator ","
set output "kalman/sample/kf_2x1x1_delayed_target_tracking_unit/kf_2x1x1_delayed_target_tracking_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_2x1x1_delayed_target_tracking_unit.csv"

# Each row is the arrival of a measurement, every 0.1 s: the estimates of both
# filters once the measurement is fused, the measurement at the instant it
# describes, one delay earlier.
set multiplot layout 3,1

# A second of the sway, where the relative velocity peaks near 8 m/s: the late
# estimate takes each measurement for the current position, a delay to the
# right of the instant it describes.
set title "{/:Bold Sample 2x1x1 Delayed Target Tracking}\nRelative Position, Peak Relative Velocity"
set xlabel "Time (s)"
set xrange [18.3:19.3]
set ylabel "Position (m)"
set autoscale y
plot DATA using 1:4 with lines linewidth 1 dashtype 2 linecolor rgb "#333333" title "True Position", \
  DATA using 2:3 with points pointtype 7 pointsize 0.8 linecolor rgb "#999999" title "Camera, at Its Instant", \
  DATA using 1:5 with linespoints linewidth 2 pointtype 5 linecolor 1 title "Late Estimate", \
  DATA using 1:6 with linespoints linewidth 2 pointtype 9 linecolor 2 title "Rewound Estimate"

set xrange [0:60]

# The late estimate lags by the distance the target moves during the delay,
# far beyond the standard deviation it claims, which ignores the delay.
set title "{/:Bold Sample 2x1x1 Delayed Target Tracking}\nPosition Error"
set ylabel "Estimate - True Position (m)"
set yrange [-0.3:1]
plot DATA using 1:(-3 * $8):(3 * $8) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Rewound ±3σ", \
  DATA using 1:(3 * $7) with lines dashtype 2 linecolor 1 title "Late +3σ", \
  DATA using 1:($5 - $4) with lines linewidth 2 linecolor 1 title "Late", \
  DATA using 1:($6 - $4) with lines linewidth 2 linecolor 2 title "Rewound"

# The late velocity estimate lags too, by about the relative velocity change
# during the delay: the vehicle's acceleration times the delay.
set title "{/:Bold Sample 2x1x1 Delayed Target Tracking}\nVelocity Error"
set ylabel "Estimate - True Velocity (m/s)"
set yrange [-0.5:0.5]
plot DATA using 1:(0.1 * $12) with lines dashtype 2 linewidth 2 linecolor rgb "#333333" title "Acceleration × Delay", \
  DATA using 1:($10 - $9) with lines linewidth 2 linecolor 1 title "Late", \
  DATA using 1:($11 - $9) with lines linewidth 2 linecolor 2 title "Rewound"

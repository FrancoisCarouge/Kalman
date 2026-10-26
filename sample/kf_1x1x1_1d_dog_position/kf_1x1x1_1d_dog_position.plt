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
set output "kalman/sample/kf_1x1x1_1d_dog_position/kf_1x1x1_1d_dog_position.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_1x1x1_1d_dog_position.csv"

set multiplot layout 3,1

set xlabel "Step"
set xrange [-0.2:10.2]
set xtics 1

# The first row is the initialization. Each step predicts the dog one meter
# further, then corrects the prediction toward the measurement: the prediction
# is the measurement less the innovation, Z - Y.
set title "{/:Bold Sample 1x1x1 1D Dog Position}\nPosition"
set ylabel "Position (m)"
set yrange [-2:20]
plot DATA using 0:($7 - sqrt($2)):($7 + sqrt($2)) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using ($0 == 0 ? NaN : $0):9:(sqrt($4)) with yerrorbars pointtype 7 title "Measurement Z ±σ", \
  DATA using ($0 == 0 ? NaN : $0):($9 - $8) with points pointtype 6 pointsize 1.5 title "Prediction", \
  DATA using 0:7 with linespoints linewidth 2 pointtype 5 title "Estimated State X"

# The prediction adds the process variance Q, the update removes part of it:
# the variances settle where both balance, the predicted variance equal to the
# sensor variance R, so the gain settles at one half.
set title "{/:Bold Sample 1x1x1 1D Dog Position}\nVariance: Prediction and Update"
set ylabel "Variance (m^2)"
set yrange [0:5]
plot DATA using 0:4 with lines dashtype 2 linewidth 2 title "Sensor Variance R", \
  DATA using ($0 == 0 ? NaN : $0):($5 - $4) with linespoints linewidth 2 pointtype 6 title "Predicted Variance P + Q", \
  DATA using ($0 == 0 ? NaN : $0):2 with linespoints linewidth 2 pointtype 5 title "Updated Variance P",

# The innovation, the measurement less its prediction, against its predicted
# standard deviation √S: the dog runs ahead of its prediction at the ninth
# step.
set title "{/:Bold Sample 1x1x1 1D Dog Position}\nInnovation"
set ylabel "Innovation (m)"
set yrange [-5:5]
plot DATA using ($0 == 0 ? NaN : $0):(-sqrt($5)):(sqrt($5)) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±√S", \
  DATA using ($0 == 0 ? NaN : $0):8 with impulses linewidth 3 linecolor rgb "#009e73" title "Innovation Y", \
  DATA using ($0 == 0 ? NaN : $0):8 with points pointtype 7 linecolor rgb "#009e73" notitle

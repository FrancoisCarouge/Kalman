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
set output "kalman/sample/kf_6x2x0_2d_vehicle_position/kf_6x2x0_2d_vehicle_position.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_6x2x0_2d_vehicle_position.csv"

set multiplot layout 3,1

set title "{/:Bold Sample 6x2x0 2D Vehicle Position}\nTrajectory, Top View"
set xlabel "X (m)"
set ylabel "Y (m)"
set xrange [-420:330]
set yrange [-20:340]
set size ratio -1
plot DATA using 2:3 with points pointtype 7 pointsize 0.6 title "Measurement Z", \
  DATA using 4:7 with lines linewidth 2 title "Estimated Position"
set size noratio

set xlabel "Measurement"
set xrange [1:35]

# Each innovation by its standard deviation √S. The initial guess, at the
# origin, lies some 500 m from the vehicle, far beyond its 22 m standard
# deviation: the first innovations are inconsistent, until the estimate
# converges. The turn, from the seventeenth measurement, surprises the constant
# acceleration model again.
set title "{/:Bold Sample 6x2x0 2D Vehicle Position}\nInnovations, Normalized"
set ylabel "Innovation / √S"
set yrange [-12:12]
plot "+" using 1:(-2):(2) with filledcurves fillcolor rgb "#eeeeee" fillstyle solid noborder title "±2", \
  DATA using 1:($16 / $18) with linespoints linewidth 2 pointtype 7 title "X", \
  DATA using 1:($17 / $19) with linespoints linewidth 2 pointtype 7 title "Y"

# The acceleration is never measured. Through the turn, its estimate tracks the
# centripetal acceleration v²/R of a 25 m/s vehicle on a 300 m radius.
set title "{/:Bold Sample 6x2x0 2D Vehicle Position}\nAcceleration"
set ylabel "Acceleration (m/s^2)"
set yrange [-3:4]
plot DATA using 1:($6 - $12):($6 + $12) with filledcurves fillcolor rgb "#cceeff" fillstyle solid noborder title "±σ X", \
  DATA using 1:($9 - $15):($9 + $15) with filledcurves fillcolor rgb "#ffe5cc" fillstyle solid noborder title "±σ Y", \
  DATA using 1:6 with lines linewidth 2 linecolor rgb "#0072b2" title "Estimated X", \
  DATA using 1:9 with lines linewidth 2 linecolor rgb "#e69f00" title "Estimated Y", \
  DATA using 1:(sqrt($6**2 + $9**2)) with lines linewidth 2 linecolor rgb "#009e73" title "Magnitude", \
  25.**2 / 300 with lines dashtype 2 linewidth 2 linecolor rgb "#333333" title "v^2/R"

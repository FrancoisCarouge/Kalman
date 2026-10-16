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
set output "kalman/sample/ekf_9x4x0_gimbal_attitude_unit/ekf_9x4x0_gimbal_attitude_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_9x4x0_gimbal_attitude_unit.csv"

set multiplot layout 3,1

# The first two panels show the time on a logarithmic scale to show the
# convergence of the first seconds.
set xlabel "Time (s, logarithmic)"
set logscale x
set xrange [0.02:30]

set title "{/:Bold Sample 9x4x0 Gimbal Attitude}\nGyroscope Bias"
set ylabel "Bias (mrad/s)"
set yrange [-30:30]
plot DATA using 1:(5 - 1000 * $8):(5 + 1000 * $8) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ Yaw", \
  DATA using 1:(1000 * $5) with lines linewidth 2 linecolor 1 title "Roll Rate", \
  DATA using 1:(1000 * $6) with lines linewidth 2 linecolor 2 title "Pitch Rate", \
  DATA using 1:(1000 * $7) with lines linewidth 2 linecolor 3 title "Yaw Rate", \
  10 with lines dashtype 2 linecolor 1 notitle, \
  -20 with lines dashtype 2 linecolor 2 notitle, \
  5 with lines dashtype 2 linecolor 3 title "True Biases"

set title "{/:Bold Sample 9x4x0 Gimbal Attitude}\nTilt Error"
set ylabel "Error (mrad)"
set yrange [-40:40]
plot DATA using 1:(-1000 * $11):(1000 * $11) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ North", \
  DATA using 1:(1000 * $9) with lines linewidth 2 title "North", \
  DATA using 1:(1000 * $10) with lines linewidth 2 title "East"

unset logscale x
set xrange [0:30]
set xlabel "Time (s)"

set title "{/:Bold Sample 9x4x0 Gimbal Attitude}\nHeading Error"
set ylabel "Error (mrad)"
set yrange [-60:60]
plot DATA using 1:(-1000 * $4):(1000 * $4) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ", \
  DATA using 1:(1000 * $2) with points pointtype 7 pointsize 0.3 linecolor rgb "#999999" title "Compass", \
  DATA using 1:(1000 * $3) with lines linewidth 2 title "Estimate"

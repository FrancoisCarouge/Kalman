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
set output "kalman/sample/ekf_3x2x0_wind_airspeed_scale_unit/ekf_3x2x0_wind_airspeed_scale_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_3x2x0_wind_airspeed_scale_unit.csv"

set multiplot layout 3,1

# The first two panels show the time on a logarithmic scale to show the
# convergence within the first circle, the time of each update offset by a
# step.
set xlabel "Time (s, logarithmic)"
set logscale x
set xrange [0.1:300]
set arrow 1 from 62.8, graph 0 to 62.8, graph 1 nohead dashtype 2
set label 1 "First circle" at 58, graph 0.92 right

set title "{/:Bold Sample 3x2x0 Wind and Airspeed Scale}\nWind"
set ylabel "Wind (m/s)"
set yrange [0:7]
plot DATA using ($1 + 0.1):($2 - $4):($2 + $4) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ North", \
  DATA using ($1 + 0.1):2 with lines linewidth 2 linecolor 1 title "North", \
  DATA using ($1 + 0.1):3 with lines linewidth 2 linecolor 2 title "East", \
  4 with lines dashtype 2 linecolor 1 notitle, \
  3 with lines dashtype 2 linecolor 2 title "True Wind"

set title "{/:Bold Sample 3x2x0 Wind and Airspeed Scale}\nAirspeed Scale Factor"
set ylabel "Scale Factor"
set yrange [0.8:1.1]
plot DATA using ($1 + 0.1):($6 - $7):($6 + $7) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ", \
  DATA using ($1 + 0.1):6 with lines linewidth 2 title "Estimate", \
  0.9 with lines dashtype 2 linewidth 2 title "True Scale Factor"

unset logscale x
unset arrow 1
unset label 1
set xrange [0:300]
set xlabel "Time (s)"

set title "{/:Bold Sample 3x2x0 Wind and Airspeed Scale}\nAirspeed Innovation and Consistency Gate"
set ylabel "Innovation (m/s)"
set yrange [-6:36]
set label 2 "Spikes rejected" at 150, 25 center
plot DATA using 1:(-$9):9 with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±5√S Gate", \
  DATA using 1:($12 == 1 ? $8 : NaN) with points pointtype 7 pointsize 0.3 title "Fused Airspeed", \
  DATA using 1:($12 == 0 ? $8 : NaN) with points pointtype 2 pointsize 0.5 title "Rejected Airspeed"

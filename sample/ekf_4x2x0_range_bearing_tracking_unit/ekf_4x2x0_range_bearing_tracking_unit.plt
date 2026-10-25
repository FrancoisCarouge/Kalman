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

set terminal svg enhanced background rgb "white" size 720,1440
set datafile separator ","
set output "kalman/sample/ekf_4x2x0_range_bearing_tracking_unit/ekf_4x2x0_range_bearing_tracking_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_4x2x0_range_bearing_tracking_unit.csv"

set multiplot layout 3,1

set xlabel "Time (s)"
set xrange [0:100]

# The target crosses the negative x axis, where the bearing wraps from π to -π,
# at these times.
do for [crossing in "0.8 32.2 63.6 95.0"] {
  set arrow from crossing, graph 0 to crossing, graph 1 nohead dashtype 2 linecolor rgb "#888888"
}

set title "{/:Bold Sample 4x2x0 Range Bearing Tracking}\nPosition Error: Unwrapped and Raw Bearing"
set ylabel "Error (m)"
set yrange [0:24]
set label 1 "Target crossing the negative x axis" at 34, 22
plot DATA using 2:9 with lines linewidth 1.5 linecolor rgb "#c0392b" title "Raw Bearing", \
  DATA using 2:3 with lines linewidth 2 linecolor rgb "#1f6fb2" title "Unwrapped Bearing"
unset label 1

set title "{/:Bold Sample 4x2x0 Range Bearing Tracking}\nDown-Range and Cross-Range Errors"
set ylabel "Error (m)"
set yrange [-2.5:2.5]
plot DATA using 2:(-$7):7 with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ Cross-Range", \
  DATA using 2:(-$6):6 with filledcurves fillcolor rgb "#aaaaaa" fillstyle solid noborder title "±σ Down-Range", \
  DATA using 2:5 with lines linewidth 1.5 linecolor rgb "#d9661f" title "Cross-Range", \
  DATA using 2:4 with lines linewidth 1.5 linecolor rgb "#1f6fb2" title "Down-Range", \
  2.08 with lines dashtype 2 linewidth 2 linecolor rgb "#444444" title "±σ of One Bearing at 8 m", \
  -2.08 with lines dashtype 2 linewidth 2 linecolor rgb "#444444" notitle

set title "{/:Bold Sample 4x2x0 Range Bearing Tracking}\nNormalized Innovation Squared"
set ylabel "NIS (logarithmic)"
set logscale y
set yrange [0.001:1000]
plot DATA using 2:10 with points pointtype 7 pointsize 0.3 linecolor rgb "#c0392b" title "Raw Bearing", \
  DATA using 2:8 with points pointtype 7 pointsize 0.3 linecolor rgb "#1f6fb2" title "Unwrapped Bearing", \
  5.991 with lines linewidth 2.5 linecolor rgb "#000000" title "95% χ² Bound, 2 Degrees of Freedom"

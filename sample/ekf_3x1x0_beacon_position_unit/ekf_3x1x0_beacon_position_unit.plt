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
set output "kalman/sample/ekf_3x1x0_beacon_position_unit/ekf_3x1x0_beacon_position_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_3x1x0_beacon_position_unit.csv"

set multiplot layout 3,1

# The first two panels count the ranges from one on a logarithmic scale to show
# the convergence during the alignment.
set xlabel "Range (logarithmic)"
set logscale x
set xrange [1:400]
set arrow 1 from 100, graph 0 to 100, graph 1 nohead dashtype 2
set label 1 "Alignment complete" at 104, graph 0.92

set title "{/:Bold Sample 3x1x0 Beacon Position}\nHeight: the Vertical Position"
set ylabel "Height, Up (m)"
set yrange [-2:4]
set label 2 "Mirror solution\nbelow the ground" at 4.5, -1.2
plot "+" using 1:(1):(3) with filledcurves fillcolor rgb "#eeeeee" fillstyle solid noborder title "Beacon Heights", \
  DATA using ($1 + 1):(-$6 - $9):(-$6 + $9) with filledcurves fillcolor rgb "#aaaaaa" fillstyle solid noborder title "±σ", \
  DATA using ($1 + 1):(-$6) with lines linewidth 2 title "Estimated Height", \
  0.5 with lines dashtype 2 linewidth 2 title "Receiver Height"
unset label 2

set title "{/:Bold Sample 3x1x0 Beacon Position}\nHorizontal Position Error"
set ylabel "Error (m)"
set yrange [-0.6:0.6]
plot DATA using ($1 + 1):(-$7):7 with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ North", \
  DATA using ($1 + 1):13 with lines linewidth 2 title "North", \
  DATA using ($1 + 1):14 with lines linewidth 2 title "East"

unset logscale x
set xrange [0:400]
set xlabel "Range"

set title "{/:Bold Sample 3x1x0 Beacon Position}\nInnovation and Consistency Gate"
set ylabel "Innovation (m)"
set yrange [-0.6:3.6]
set label 3 "Multipath reflection rejected" at 285, 2.4 right
set arrow 2 from 287, 2.5 to 297, 3.0 linewidth 1
plot DATA using 1:(-3 * $11):(3 * $11) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±3√S Gate", \
  DATA using 1:($12 == 1 ? $10 : NaN) with points pointtype 7 pointsize 0.5 title "Fused Range", \
  DATA using 1:($12 == 0 ? $10 : NaN) with points pointtype 11 pointsize 2 title "Rejected Range"

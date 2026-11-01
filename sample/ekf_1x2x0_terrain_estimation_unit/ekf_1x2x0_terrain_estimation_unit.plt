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
set output "kalman/sample/ekf_1x2x0_terrain_estimation_unit/ekf_1x2x0_terrain_estimation_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_1x2x0_terrain_estimation_unit.csv"

set multiplot layout 3,1

set xlabel "Time (s)"
set xrange [0:60]
# The range finder dropout.
set object 1 rectangle from 20, graph 0 to 40, graph 1 fillcolor rgb "#f0f0f0" fillstyle solid noborder behind
set label 1 "Range finder dropout" at 30, graph 0.93 center

set title "{/:Bold Sample 1x2x0 Terrain Estimation}\nTerrain Altitude"
set ylabel "Altitude (m)"
set yrange [0:34]
set label 2 "Branch in the beam,\nrejected" at 46, 23 right
set arrow 1 from 46.5, 22 to 49.3, 19.6 linewidth 1
plot 30 with lines dashtype 2 linewidth 2 linecolor rgb "#1f6fb2" title "Vehicle", \
  DATA using 1:($7 == 2 ? $5 : NaN) with points pointtype 11 pointsize 2 linecolor rgb "#c0392b" title "Rejected Range", \
  DATA using 1:2 with lines dashtype 2 linewidth 2 linecolor rgb "#2f7d32" title "Terrain", \
  DATA using 1:3 with lines linewidth 2 linecolor rgb "#d9661f" title "Estimate"
unset label 2
unset arrow 1

set title "{/:Bold Sample 1x2x0 Terrain Estimation}\nTerrain Error and Uncertainty"
set ylabel "Error (m)"
set yrange [-2:2]
plot DATA using 1:(-$10):10 with filledcurves fillcolor rgb "#e3e3e3" fillstyle solid noborder title "±σ Without Flow", \
  DATA using 1:(-$4):4 with filledcurves fillcolor rgb "#aaaaaa" fillstyle solid noborder title "±σ", \
  DATA using 1:($3 - $2) with lines linewidth 2 linecolor rgb "#d9661f" title "Error"

set title "{/:Bold Sample 1x2x0 Terrain Estimation}\nNormalized Innovation and Consistency Gate"
set ylabel "Innovation (σ)"
set yrange [-4:4]
set label 2 "Branch: -58σ, rejected" at 49, -3.5 right
plot "+" using 1:(-3):(3) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±3σ Gate", \
  DATA using 1:($7 == 1 ? $6 : NaN) with points pointtype 7 pointsize 0.4 linecolor rgb "#1f6fb2" title "Range", \
  DATA using 1:($9 == 1 ? $8 : NaN) with points pointtype 7 pointsize 0.4 linecolor rgb "#d9661f" title "Flow", \
  DATA using 1:($7 == 2 ? -3.9 : NaN) with points pointtype 11 pointsize 2 linecolor rgb "#c0392b" title "Rejected Range"

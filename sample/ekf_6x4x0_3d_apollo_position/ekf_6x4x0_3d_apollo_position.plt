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
set output "kalman/sample/ekf_6x4x0_3d_apollo_position/ekf_6x4x0_3d_apollo_position.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_6x4x0_3d_apollo_position.csv"

set multiplot layout 3,1

set xlabel "Time (s)"
set xrange [1:10]

# Each innovation by its standard deviation √S: a consistent radar measurement
# lies mostly within one.
set title "{/:Bold Sample 6x4x0 Apollo Position}\nRadar Innovations, Normalized"
set ylabel "Innovation / √S"
set yrange [-2:1]
plot "+" using 1:(-1):(1) with filledcurves fillcolor rgb "#eeeeee" fillstyle solid noborder title "±1", \
  DATA using 1:($18 / $22) with linespoints linewidth 2 pointtype 7 title "Range r", \
  DATA using 1:($19 / $23) with linespoints linewidth 2 pointtype 7 title "Range Rate", \
  DATA using 1:($20 / $24) with linespoints linewidth 2 pointtype 7 title "Shaft β", \
  DATA using 1:($21 / $25) with linespoints linewidth 2 pointtype 7 title "Trunnion θ"

# The position the radar angles imply, across the line of sight, against the
# estimate initialized away from it.
set title "{/:Bold Sample 6x4x0 Apollo Position}\nCross Range Position"
set ylabel "Position (m)"
set yrange [200:1100]
plot DATA using 1:($7 - $13):($7 + $13) with filledcurves fillcolor rgb "#cceeff" fillstyle solid noborder title "±σ y", \
  DATA using 1:($8 - $14):($8 + $14) with filledcurves fillcolor rgb "#ffe5cc" fillstyle solid noborder title "±σ z", \
  DATA using 1:7 with lines linewidth 2 linecolor rgb "#0072b2" title "Estimated y", \
  DATA using 1:8 with lines linewidth 2 linecolor rgb "#e69f00" title "Estimated z", \
  DATA using 1:($2 * cos($5) * sin($4)) with points pointtype 7 linecolor rgb "#0072b2" title "Radar y", \
  DATA using 1:($2 * sin($5)) with points pointtype 7 linecolor rgb "#e69f00" title "Radar z"

# The constant velocity model reads the cross range offset partly as motion.
set title "{/:Bold Sample 6x4x0 Apollo Position}\nCross Range Velocity"
set ylabel "Velocity (m/s)"
set yrange [-25:15]
plot DATA using 1:($10 - $16):($10 + $16) with filledcurves fillcolor rgb "#cceeff" fillstyle solid noborder title "±σ vy", \
  DATA using 1:($11 - $17):($11 + $17) with filledcurves fillcolor rgb "#ffe5cc" fillstyle solid noborder title "±σ vz", \
  DATA using 1:10 with lines linewidth 2 linecolor rgb "#0072b2" title "Estimated vy", \
  DATA using 1:11 with lines linewidth 2 linecolor rgb "#e69f00" title "Estimated vz", \
  0 with lines dashtype 2 linewidth 2 linecolor rgb "#666666" title "Straight Approach"

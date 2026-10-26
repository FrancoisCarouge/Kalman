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
set output "kalman/sample/kf_8x4x0_2d_bounding_box/kf_8x4x0_2d_bounding_box.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_8x4x0_2d_bounding_box.csv"

set multiplot layout 3,1

set xlabel "Frame"
set xrange [1:100]

# The pedestrian walks left, back right, left again, then turns around.
set title "{/:Bold Sample 8x4x0 2D Bounding Box}\nCenter: Horizontal Position"
set ylabel "Center x (px)"
set yrange [280:660]
plot DATA using 1:($6 - $14):($6 + $14) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using 1:2 with points pointtype 7 pointsize 0.6 title "Detection", \
  DATA using 1:6 with lines linewidth 2 title "Estimated Center x"

# The uncertainties scale with the height of the box: a taller, closer box
# moves more pixels for the same motion.
set title "{/:Bold Sample 8x4x0 2D Bounding Box}\nHeight"
set ylabel "Height h (px)"
set yrange [240:400]
plot DATA using 1:($9 - $15):($9 + $15) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using 1:5 with points pointtype 7 pointsize 0.6 title "Detection", \
  DATA using 1:9 with lines linewidth 2 title "Estimated Height h"

# Deep SORT gates the association of a detection to a track by the squared
# Mahalanobis distance of the innovation, against the 95% quantile of the
# chi-square distribution of four degrees of freedom. Its mean, four for a
# consistent filter, is less than one: the noise model is conservative. Only
# the jump of the box height at the tenth frame falls beyond the gate.
set title "{/:Bold Sample 8x4x0 2D Bounding Box}\nAssociation Gate"
set ylabel "Squared Mahalanobis Distance"
set yrange [0:22]
plot 9.4877 with lines dashtype 2 linewidth 2 linecolor rgb "#c0392b" title "95% Gate χ^2_4", \
  4 with lines dashtype 3 linewidth 2 linecolor rgb "#333333" title "Consistent Mean", \
  DATA using 1:17 with impulses linewidth 2 linecolor rgb "#0072b2" title "y^T S^{-1} y"

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
set output "kalman/sample/ekf_4x1x0_thermal_soaring/ekf_4x1x0_thermal_soaring.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_4x1x0_thermal_soaring.csv"

set multiplot layout 3,1

set xlabel "Variometer Measurement"
set xrange [1:101]

# The sUAV flies within a meter of its origin, near the thermal center: the
# variometer measures the strength W almost directly, at most 0.45 m/s apart.
set title "{/:Bold Sample 4x1x0 Thermal Soaring}\nStrength: the Updraft at the Thermal Center"
set ylabel "Vertical Air Velocity (m/s)"
set yrange [-0.2:1.2]
plot DATA using ($1 + 1):($3 - $5):($3 + $5) with filledcurves fillcolor rgb "#eeeeee" fillstyle solid noborder title "Predicted ±√S", \
  DATA using ($1 + 1):($6 - $10):($6 + $10) with filledcurves fillcolor rgb "#aaaaaa" fillstyle solid noborder title "±σ", \
  DATA using ($1 + 1):2 with points pointtype 7 pointsize 0.5 title "Variometer", \
  DATA using ($1 + 1):6 with lines linewidth 2 title "Estimated Strength W", \
  0.5026 with lines dashtype 2 linewidth 2 title "Mean Variometer"

# Each prediction drifts the thermal with the wind, while each update draws the
# center back toward the sUAV, where the measured updraft is stronger than
# predicted.
set title "{/:Bold Sample 4x1x0 Thermal Soaring}\nThermal Center: Wind Drift and Correction"
set ylabel "Distance from the sUAV (m)"
set yrange [0:60]
plot DATA using ($1 + 1):(north = (column(1) == 0 ? 5 : north) + $14) with lines dashtype 2 linewidth 2 linecolor rgb "#009e73" title "Drift Only North", \
  DATA using ($1 + 1):(east = (column(1) == 0 ? 0 : east) + $15) with lines dashtype 2 linewidth 2 linecolor rgb "#56b4e9" title "Drift Only East", \
  DATA using ($1 + 1):8 with lines linewidth 2 linecolor rgb "#009e73" title "Estimated North", \
  DATA using ($1 + 1):9 with lines linewidth 2 linecolor rgb "#56b4e9" title "Estimated East"

# A standard deviation relative to its initial value: the measurements inform
# the strength, but barely the radius and the center, from so close.
set title "{/:Bold Sample 4x1x0 Thermal Soaring}\nObservability: the Uncertainty Remaining"
set ylabel "σ / Initial σ (%)"
set yrange [50:105]
plot DATA using ($1 + 1):(100 * $10 / 0.07) with lines linewidth 2 title "Strength W", \
  DATA using ($1 + 1):(100 * $11 / 20) with lines linewidth 2 title "Radius R", \
  DATA using ($1 + 1):(100 * $12 / 20) with lines linewidth 2 title "North x", \
  DATA using ($1 + 1):(100 * $13 / 20) with lines linewidth 2 title "East y"

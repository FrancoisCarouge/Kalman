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
set output "kalman/sample/kf_2x1x1_1d_rocket_altitude/kf_2x1x1_1d_rocket_altitude.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_2x1x1_1d_rocket_altitude.csv"

# The reference trajectory: a boost of 30 m.s^-2 from rest at the first
# measurement, the time t of a measurement n being 0.25 (n - 1) s, which the
# altimeter measurements fit within their standard deviation.
elapsed(n) = 0.25 * (n - 1)
reference_altitude(n) = 15 * elapsed(n)**2
reference_velocity(n) = 30 * elapsed(n)

set multiplot layout 3,1

set xlabel "Time (s)"
set xrange [-0.1:7.35]

# The errors from the reference trajectory, the altitude climbing to 750 m.
set title "{/:Bold Sample 2x1x1 1D Rocket Altitude}\nAltitude Error"
set ylabel "Altitude - Reference (m)"
set yrange [-80:60]
plot DATA using (elapsed($1)):($3 - reference_altitude($1) - sqrt($5)):($3 - reference_altitude($1) + sqrt($5)) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using (elapsed($1)):($2 - reference_altitude($1)):(20) with yerrorbars pointtype 7 title "Altimeter ±σ", \
  DATA using (elapsed($1)):($3 - reference_altitude($1)) with linespoints linewidth 2 pointtype 5 title "Estimated Altitude"

# The velocity is not measured: the altitude measurements and the
# accelerometer input reveal it, from an initial guess at rest.
set title "{/:Bold Sample 2x1x1 1D Rocket Altitude}\nVelocity Error"
set ylabel "Velocity - Reference (m/s)"
set yrange [-30:20]
plot DATA using (elapsed($1)):($4 - reference_velocity($1) - sqrt($6)):($4 - reference_velocity($1) + sqrt($6)) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using (elapsed($1)):($4 - reference_velocity($1)) with linespoints linewidth 2 pointtype 5 title "Estimated Velocity", \
  0 with lines dashtype 2 linewidth 2 linecolor rgb "#333333" notitle

# The accurate accelerometer input carries the prediction: the gains fall, the
# altitude gain to 0.12, the estimate weighing more than the measurement.
set title "{/:Bold Sample 2x1x1 1D Rocket Altitude}\nGain"
set ylabel "Gain"
set yrange [0:0.6]
plot DATA using (elapsed($1)):7 with linespoints linewidth 2 pointtype 5 title "Altitude Gain", \
  DATA using (elapsed($1)):8 with linespoints linewidth 2 pointtype 7 title "Velocity Gain (s^{-1})"

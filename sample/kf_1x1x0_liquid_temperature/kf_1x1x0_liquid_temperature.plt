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
set output "kalman/sample/kf_1x1x0_liquid_temperature/kf_1x1x0_liquid_temperature.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_1x1x0_liquid_temperature.csv"

# The true temperature at the measurement points.
array truth[10] = [49.979, 50.025, 50., 50.003, 49.994, 50.002, 49.999, 50.006, 49.998, 49.991]

# The steady state the process noise q = 0.0001 °C² sustains against the
# thermometer variance r = 0.01 °C²: the estimate variance p = (√(q² + 4qr) -
# q) / 2, and the gain k = (p + q) / (p + q + r).
steady_p = (sqrt(0.0001**2 + 4 * 0.0001 * 0.01) - 0.0001) / 2
steady_k = (steady_p + 0.0001) / (steady_p + 0.0001 + 0.01)

set multiplot layout 3,1

set xlabel "Measurement Step"
set xrange [0.8:10.2]
set xtics 1

# The estimate starts from a 10 °C guess, outside of the view, before the
# first measurement.
set title "{/:Bold Sample 1x1x0 Liquid Temperature}\nTemperature"
set ylabel "Temperature (°C)"
set yrange [49.7:50.3]
plot DATA using ($0 == 0 ? NaN : $0):($6 - sqrt($2)):($6 + sqrt($2)) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using ($0 == 0 ? NaN : $0):8:(0.1) with yerrorbars pointtype 7 title "Measurement Z ±σ", \
  DATA using ($0 == 0 ? NaN : $0):($0 == 0 ? NaN : truth[int($0)]) with linespoints dashtype 2 linewidth 2 pointtype 6 linecolor rgb "#333333" title "True Temperature", \
  DATA using ($0 == 0 ? NaN : $0):6 with linespoints linewidth 2 pointtype 5 title "Estimated State X"

set title "{/:Bold Sample 1x1x0 Liquid Temperature}\nUncertainty and Error"
set ylabel "Standard Deviation, Error (°C)"
set yrange [0:0.12]
plot DATA using 0:(sqrt($4)) with lines dashtype 2 linewidth 2 title "Thermometer σ = √R", \
  sqrt(steady_p) with lines dashtype 3 linewidth 2 title "Steady State σ", \
  DATA using ($0 == 0 ? NaN : $0):(sqrt($2)) with linespoints linewidth 2 pointtype 5 title "Estimate σ = √P", \
  DATA using ($0 == 0 ? NaN : $0):($0 == 0 ? NaN : abs($6 - truth[int($0)])) with linespoints linewidth 2 pointtype 7 title "Absolute Error |X - T|"

# Unlike an average, whose gain falls as 1/n toward zero, the process noise
# floors the gain at its steady state: the estimate keeps following the
# fluctuations.
set title "{/:Bold Sample 1x1x0 Liquid Temperature}\nGain"
set ylabel "Gain K"
set yrange [0:1.05]
plot DATA using ($0 == 0 ? NaN : $0):1 with linespoints linewidth 2 pointtype 5 title "Gain K", \
  DATA using ($0 == 0 ? NaN : $0):(1 / $0) with lines dashtype 2 linewidth 2 title "Average Weight 1/n", \
  steady_k with lines dashtype 3 linewidth 2 title "Steady State Gain"

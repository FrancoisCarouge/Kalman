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
set output "kalman/sample/kf_1x1x0_building_height/kf_1x1x0_building_height.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/kf_1x1x0_building_height.csv"

set multiplot layout 3,1

set xlabel "Measurement Step"
set xrange [-0.2:10.2]
set xtics 1

# The first row is the initialization, before any measurement. With a constant
# height and no process noise, the estimate weighs the initial guess as one
# more, less precise, measurement: it nearly follows the running mean.
set title "{/:Bold Sample 1x1x0 Building Height}\nHeight"
set ylabel "Height (m)"
set yrange [30:80]
plot DATA using 0:($8 - sqrt($4)):($8 + sqrt($4)) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Estimate ±σ", \
  DATA using ($0 == 0 ? NaN : $0):10:(5) with yerrorbars pointtype 7 title "Measurement Z ±σ", \
  DATA using 0:(total = ($0 == 0 ? 0 : total + $10), $0 == 0 ? NaN : total / $0) with lines dashtype 2 linewidth 2 title "Running Mean", \
  DATA using 0:8 with linespoints linewidth 2 pointtype 5 title "Estimated State X", \
  50 with lines linewidth 2 linecolor rgb "#333333" title "True Height"

# The standard deviation of the estimate falls as one over the square root of
# the number of measurements, below the altimeter's 5 m, while the error of a
# given estimate scatters below it.
set title "{/:Bold Sample 1x1x0 Building Height}\nUncertainty and Error"
set ylabel "Standard Deviation, Error (m)"
set yrange [0:16]
plot DATA using 0:(sqrt($6)) with lines dashtype 2 linewidth 2 title "Altimeter σ = √R", \
  DATA using 0:(sqrt($4)) with linespoints linewidth 2 pointtype 5 title "Estimate σ = √P", \
  DATA using 0:(abs($8 - 50)) with linespoints linewidth 2 pointtype 7 title "Absolute Error |X - 50|"

# The gain weighs each measurement: close to one over the number of
# measurements, as an average does, once the initial guess weighs little.
set title "{/:Bold Sample 1x1x0 Building Height}\nGain"
set ylabel "Gain K"
set yrange [0:1]
plot DATA using ($0 == 0 ? NaN : $0):3 with linespoints linewidth 2 pointtype 5 title "Gain K", \
  DATA using ($0 == 0 ? NaN : $0):(1 / $0) with lines dashtype 2 linewidth 2 title "Average Weight 1/n"

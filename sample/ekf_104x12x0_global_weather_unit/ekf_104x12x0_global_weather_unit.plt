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
set output "kalman/sample/ekf_104x12x0_global_weather_unit/ekf_104x12x0_global_weather_unit.svg"
set timestamp
set grid
set key bmargin center horizontal

DATA = "/tmp/kalman/ekf_104x12x0_global_weather_unit.csv"

set multiplot layout 3,1

set xlabel "Time (day)"
set xrange [0:10]

set title "{/:Bold Sample 104x12x0 Global Weather}\nLower Temperature Errors: Sounded, Neighbouring, and Beyond the Localization Cutoff"
set ylabel "Error (K)"
set yrange [-3:3]
plot DATA using 1:(-$7):7 with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ Beyond the Cutoff", \
  DATA using 1:(-$5):5 with filledcurves fillcolor rgb "#bbbbbb" fillstyle solid noborder title "±σ Neighbouring", \
  DATA using 1:(-$3):3 with filledcurves fillcolor rgb "#888888" fillstyle solid noborder title "±σ Sounded", \
  DATA using 1:6 with linespoints pointtype 7 pointsize 0.4 linewidth 1.5 linecolor rgb "#2a9d4b" title "90°W, 45°N", \
  DATA using 1:4 with linespoints pointtype 7 pointsize 0.4 linewidth 1.5 linecolor rgb "#d9661f" title "180°, 45°N", \
  DATA using 1:2 with linespoints pointtype 7 pointsize 0.4 linewidth 1.5 linecolor rgb "#1f6fb2" title "90°E, 45°N"

set title "{/:Bold Sample 104x12x0 Global Weather}\nSurface Pressure Errors: Station and Unobserved"
set ylabel "Error (Pa)"
set yrange [-400:400]
plot DATA using 1:(-$11):11 with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "±σ Unobserved", \
  DATA using 1:(-$9):9 with filledcurves fillcolor rgb "#888888" fillstyle solid noborder title "±σ Station", \
  DATA using 1:10 with linespoints pointtype 7 pointsize 0.4 linewidth 1.5 linecolor rgb "#d9661f" title "90°W, 45°N", \
  DATA using 1:8 with linespoints pointtype 7 pointsize 0.4 linewidth 1.5 linecolor rgb "#1f6fb2" title "0°, 45°N"

set title "{/:Bold Sample 104x12x0 Global Weather}\nError of the 104 States, Each in Units of Its Hourly Model Error"
set ylabel "Root mean square"
set yrange [0:40]
plot DATA using 1:13 with linespoints pointtype 7 pointsize 0.4 linewidth 1.5 linecolor rgb "#c0392b" title "Climate", \
  DATA using 1:12 with linespoints pointtype 7 pointsize 0.4 linewidth 2 linecolor rgb "#1f6fb2" title "Filter Estimate"

unset multiplot

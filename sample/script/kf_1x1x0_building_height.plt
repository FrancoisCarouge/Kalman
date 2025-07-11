#!/usr/bin/gnuplot
#  _  __          _      __  __          _   _
# | |/ /    /\   | |    |  \/  |   /\   | \ | |
# | ' /    /  \  | |    | \  / |  /  \  |  \| |
# |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
# | . \  / ____ \| |____| |  | |/ ____ \| |\  |
# |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

# Kalman Filter
# Version 0.5.3
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

set terminal svg enhanced background rgb "white" size 720,1080
set datafile separator ","
set output "kalman/sample/image/kf_1x1x0_building_height.svg"
set timestamp
set ylabel "Height (m)"
set xlabel "Measurement Step"
set grid ytics
set xtics 1
set key bmargin center horizontal

set multiplot layout 3,1

set title "{/:Bold Sample 1x1x0 Building Height}\nStates"
plot "/tmp/kalman/kf_1x1x0_building_height.csv" using ($0):10 with linespoints linewidth 3 pointtype 5 title "Measurement Output Z", \
  "/tmp/kalman/kf_1x1x0_building_height.csv" using ($0):8 with linespoints linewidth 3 pointtype 5 title "Estimated State X", \
  50 with lines linewidth 3 title "True State"

set title "{/:Bold Sample 1x1x0 Building Height}\nUncertainties"
plot "/tmp/kalman/kf_1x1x0_building_height.csv" using ($0):6 with linespoints linewidth 3 pointtype 5 title "Measurement Uncertainty R", \
  "/tmp/kalman/kf_1x1x0_building_height.csv" using ($0):4 with linespoints linewidth 3 pointtype 5 title "Estimation Uncertainty P"

set title "{/:Bold Sample 1x1x0 Building Height}\nGain"
plot "/tmp/kalman/kf_1x1x0_building_height.csv" using ($0):3 with linespoints linewidth 3 pointtype 5 title "Gain K"

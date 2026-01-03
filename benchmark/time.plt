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

set terminal png enhanced background rgb "white" size 1600,1200
set datafile separator ","
set title "{/:Bold KALMAN FILTER PREDICTION TIME}\n{Backend: Eigen}\n{Representation:${REPRESENTATION}}\n{State x Output}" offset 20,0
set output "time_predict_${REPRESENTATION}_state_output.png"
set timestamp
set grid
set grid ztics
set grid vertical layerdefault
set xlabel "State (count)"
set ylabel "Output (count)"
set zlabel "Time (ns)"
set zlabel rotate by 90
set ticslevel 0
set xtics 1
set ytics 1
set view 40,320,1,1
set pm3d
set key noautotitle
set boxdepth 1
splot "time.csv" using 1:2:(((strcol(4) eq "predict") && (strcol(5) eq "${REPRESENTATION}"))?($6 * 1000000000):1/0) with boxes lc palette
unset output

set terminal png enhanced background rgb "white" size 1600,1200
set datafile separator ","
set title "{/:Bold Eigen Predict ${REPRESENTATION} State x Input Benchmark}\n"
set output "time_predict_${REPRESENTATION}_state_input.png"
set timestamp
set grid
set grid ztics
set grid vertical layerdefault
set xlabel "State (count)"
set ylabel "Input (count)"
set zlabel "Time (ns)"
set zlabel rotate by 90
set ticslevel 0
set xtics 1
set ytics 1
set view 40,320,1,1
set pm3d
set key noautotitle
set boxdepth 1
splot "time.csv" using 1:3:(((strcol(4) eq "predict") && (strcol(5) eq "${REPRESENTATION}"))?($6 * 1000000000):1/0) with boxes lc palette
unset output

set terminal png enhanced background rgb "white" size 1600,1200
set datafile separator ","
set title "{/:Bold Eigen Update ${REPRESENTATION} State x Output Benchmark}\n"
set output "time_update_${REPRESENTATION}_state_output.png"
set timestamp
set grid
set grid ztics
set grid vertical layerdefault
set xlabel "State (count)"
set ylabel "Output (count)"
set zlabel "Time (ns)"
set zlabel rotate by 90
set ticslevel 0
set xtics 1
set ytics 1
set view 40,320,1,1
set pm3d
set key noautotitle
set boxdepth 1
splot "time.csv" using 1:2:(((strcol(4) eq "update") && (strcol(5) eq "${REPRESENTATION}"))?($6 * 1000000000):1/0) with boxes lc palette
unset output

set terminal png enhanced background rgb "white" size 1600,1200
set datafile separator ","
set title "{/:Bold Eigen Update ${REPRESENTATION} State x Input Benchmark}\n"
set output "time_update_${REPRESENTATION}_state_input.png"
set timestamp
set grid
set grid ztics
set grid vertical layerdefault
set xlabel "State (count)"
set ylabel "Input (count)"
set zlabel "Time (ns)"
set zlabel rotate by 90
set ticslevel 0
set xtics 1
set ytics 1
set view 40,320,1,1
set pm3d
set key noautotitle
set boxdepth 1
splot "time.csv" using 1:3:(((strcol(4) eq "update") && (strcol(5) eq "${REPRESENTATION}"))?($6 * 1000000000):1/0) with boxes lc palette
unset output

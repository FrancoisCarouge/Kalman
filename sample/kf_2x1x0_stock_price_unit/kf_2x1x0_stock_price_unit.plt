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

set terminal svg enhanced background rgb "white" size 720,720
set datafile separator ","
set output "kalman/sample/kf_2x1x0_stock_price_unit/kf_2x1x0_stock_price_unit.svg"
set timestamp
set xlabel "Trading Day"
set grid ytics
set key bmargin center horizontal

set multiplot layout 2,1

set title "{/:Bold Sample 2x1x0 Stock Price}\nNext Day Forecast"
set ylabel "Closing Price ($)"
set yrange [86:108]
plot "/tmp/kalman/kf_2x1x0_stock_price_unit.csv" using ($1 + 1):($5 - $6):($5 + $6) with filledcurves fillcolor rgb "#dddddd" fillstyle solid noborder title "Forecast ±1σ", \
  "/tmp/kalman/kf_2x1x0_stock_price_unit.csv" using ($1 + 1):5 with lines linewidth 3 title "Forecast", \
  "/tmp/kalman/kf_2x1x0_stock_price_unit.csv" using 1:3 with lines linewidth 3 title "Estimated Level", \
  "/tmp/kalman/kf_2x1x0_stock_price_unit.csv" using 1:2 with points pointtype 7 title "Measured Closing Price"

set title "{/:Bold Sample 2x1x0 Stock Price}\nTrend"
set ylabel "Slope ($/day)"
set yrange [*:*]
plot "/tmp/kalman/kf_2x1x0_stock_price_unit.csv" using 1:4 with linespoints linewidth 3 pointtype 5 title "Estimated Slope", \
  0 with lines linewidth 1 linecolor "black" notitle

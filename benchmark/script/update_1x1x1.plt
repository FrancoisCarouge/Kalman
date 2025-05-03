#!/usr/bin/gnuplot
#  _  __          _      __  __          _   _
# | |/ /    /\   | |    |  \/  |   /\   | \ | |
# | ' /    /  \  | |    | \  / |  /  \  |  \| |
# |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
# | . \  / ____ \| |____| |  | |/ ____ \| |\  |
# |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

# Kalman Filter
# Version 0.5.1
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

set terminal svg enhanced background rgb "white" size 360,720
set datafile separator ","
set title "{/:Bold Update 1x1x1 Float Benchmark}\n{/*0.8 kalman<float, float, float>::update(0.f)}"
set output "kalman/benchmark/image/update_1x1x1.svg"
set timestamp
set key noautotitle
set key inside top left reverse Left
set ylabel "Time (ns)"
set grid ytics
set boxwidth 0.9
set xrange [ -0.5 : 0.5 ]
set style fill solid border linecolor "black"
set yrange [24 : 27]
set ytics .2
set xtics ("Update 1x1x1 Float" 0)

plot "/tmp/kalman/update_1x1x1.csv" using (0):6 with boxes linecolor rgb "#F7DC6F" title "Maximum", \
  "" using (0):1 with boxes linecolor rgb "#F4D03F" title "Average", \
  "" using (0):5 with boxes linecolor rgb "#F1C40F" title "Minimum", \
  "" using (0):6:(sprintf("%8.2f", $6)) with labels right offset char -2,0.3, \
  "" using (0):1:(sprintf("%8.2f", $1)) with labels right offset char -2,0.3, \
  "" using (0):5:(sprintf("%8.2f", $5)) with labels right offset char -2,0.3, \
  "" using (0):2:3 with yerrorbars linetype 1 linecolor "black" title "Mean and Standard Deviation", \
  "" using (0):($2+$3):(sprintf("%.2f", $2+$3)) with labels left offset char 1,0, \
  "" using (0):2:(sprintf("%.2f (cv: %.1f %%)", $2, $4 * 100)) with labels left offset char 1,0, \
  "" using (0):($2-$3):(sprintf("%.2f", $2-$3)) with labels left offset char 1,0

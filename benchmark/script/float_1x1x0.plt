#!/usr/bin/gnuplot
#  _  __          _      __  __          _   _
# | |/ /    /\   | |    |  \/  |   /\   | \ | |
# | ' /    /  \  | |    | \  / |  /  \  |  \| |
# |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
# | . \  / ____ \| |____| |  | |/ ____ \| |\  |
# |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

# Kalman Filter for C++
# Version 0.1.0
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
set title "{/:Bold Float 1x1x0 Group Benchmark}\n"
set output "kalman/benchmark/image/float_1x1x0.svg"
set ylabel "Time (ns)"
set grid ytics
set boxwidth 0.9
set xrange [ -0.5 : 2.5 ]
set style fill solid border linecolor "black"
set yrange [12 : 27]
set ytics 1
set xtics ("Baseline - No Code" 0, "Predict" 1, "Update" 2) rotate by 345
set key noautotitle

plot "/tmp/kalman/baseline.csv" using (0):1 with boxes linecolor rgb "#F7DC6F", \
  "/tmp/kalman/predict_1x1x0.csv" using (1):1 with boxes linecolor rgb "#F4D03F", \
  "/tmp/kalman/update_1x1x0.csv" using (2):1 with boxes linecolor rgb "#F4D03F"

#!/bin/bash
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

set -e

# Filter benchmark results to flatten and retain plot data.
jq --compact-output '[.benchmarks[]
  | select(has("aggregate_name"))
  | select(has("big_o") | not)
  | select(has("rms") | not)
  | {name: .run_name, value: {(.aggregate_name): .real_time}}]
  | group_by([.name])
  | map((.[0]|del(.value)) + { values: (map(.value)) })[]
  | {name: .name, mean: .values[0].mean, median: .values[1].median, stddev: .values[2].stddev, cv: .values[3].cv, min: .values[4].min, max: .values[5].max}
  ' results.json > /tmp/flat_results.json

# Individual CSV and plot results.
grep "baseline" /tmp/flat_results.json \
  | sed -E 's#\{"name":"baseline/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/baseline.csv
gnuplot baseline.plt

grep "predict1x1x0" /tmp/flat_results.json \
  | sed -E 's#\{"name":"predict1x1x0/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/predict1x1x0.csv
gnuplot predict1x1x0.plt

grep "update1x1x0" /tmp/flat_results.json \
  | sed -E 's#\{"name":"update1x1x0/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/update1x1x0.csv
gnuplot update1x1x0.plt

grep "operator1x1x0" /tmp/flat_results.json \
  | sed -E 's#\{"name":"operator1x1x0/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/operator1x1x0.csv
gnuplot operator1x1x0.plt

grep "predict1x1x1" /tmp/flat_results.json \
  | sed -E 's#\{"name":"predict1x1x1/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/predict1x1x1.csv
gnuplot predict1x1x1.plt

grep "update1x1x1" /tmp/flat_results.json \
  | sed -E 's#\{"name":"update1x1x1/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/update1x1x1.csv
gnuplot update1x1x1.plt

grep "operator1x1x1" /tmp/flat_results.json \
  | sed -E 's#\{"name":"operator1x1x1/repeats:100/manual_time","mean":([0-9.]*),"median":([0-9.]*),"stddev":([0-9.]*),"cv":([0-9.]*),"min":([0-9.]*),"max":([0-9.]*)}#\1, \2, \3, \4, \5, \6#' \
  > /tmp/operator1x1x1.csv
gnuplot operator1x1x1.plt

# Groups using results.
gnuplot float1x1x0.plt
gnuplot float1x1x1.plt
gnuplot float.plt

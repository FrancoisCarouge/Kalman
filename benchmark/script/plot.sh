#!/bin/bash
#  _  __          _      __  __          _   _
# | |/ /    /\   | |    |  \/  |   /\   | \ | |
# | ' /    /  \  | |    | \  / |  /  \  |  \| |
# |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
# | . \  / ____ \| |____| |  | |/ ____ \| |\  |
# |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

# Kalman Filter
# Version 0.2.0
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

mv build/benchmark/*.json kalman/benchmark/result 2>/dev/null || true

rm -rf /tmp/kalman
mkdir /tmp/kalman

RESULTS=`find "kalman/benchmark/result" -iname "*.json"`
for RESULT in ${RESULTS}; do
  NAME=$(basename ${RESULT} .json)
  jq --compact-output '[.benchmarks[]
  | select(has("aggregate_name"))
  | select(has("big_o") | not)
  | select(has("rms") | not)
  | {name: .run_name, value: {(.aggregate_name): .real_time}}]
  | group_by([.name])
  | map((.[0]|del(.value)) + { values: (map(.value)) })[]
  | {name: .name, mean: .values[0].mean, median: .values[1].median, stddev: .values[2].stddev, cv: .values[3].cv, min: .values[4].min, max: .values[5].max}
  ' ${RESULT} > /tmp/kalman/${NAME}.json
  grep "manual_time" /tmp/kalman/${NAME}.json \
    | sed -E 's#\{"name":".*/repeats:[0-9]*/manual_time","mean":(.*),"median":(.*),"stddev":(.*),"cv":(.*),"min":(.*),"max":(.*)}#\1, \2, \3, \4, \5, \6#' \
    > /tmp/kalman/${NAME}.csv
done

for STATE in {1..32}; do
  for OUTPUT in {1..32}; do
    echo -n "${STATE}, ${OUTPUT}, " >> "/tmp/kalman/eigen_update.csv"
    cat "/tmp/kalman/eigen_update_${STATE}x${OUTPUT}x0.csv" >> "/tmp/kalman/eigen_update.csv"
  done
  for INPUT in {1..32}; do
    echo -n "${STATE}, ${INPUT}, " >> "/tmp/kalman/eigen_predict.csv"
    cat "/tmp/kalman/eigen_predict_${STATE}x1x${INPUT}.csv" >> "/tmp/kalman/eigen_predict.csv"
  done
done

PLOTS=`find "kalman/benchmark/script" -iname "*.plt"`
for PLOT in ${PLOTS}; do
  gnuplot ${PLOT}
done

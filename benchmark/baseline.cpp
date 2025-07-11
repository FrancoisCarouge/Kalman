/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> */

#include "benchmark.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>

namespace fcarouge::benchmark {
namespace {
//! @benchmark Measure baseline, empty benchmark performance.
void bench(::benchmark::State &benchmark_state) {
  for (auto _ : benchmark_state) {
    ::benchmark::ClobberMemory();
    const auto start{clock::now()};

    // Nothing to measure but the benchmark.

    ::benchmark::ClobberMemory();
    const auto end{clock::now()};

    benchmark_state.SetIterationTime(
        std::chrono::duration<double>{end - start}.count());
  }
}

BENCHMARK(bench)
    ->Name("baseline")
    ->Unit(::benchmark::kNanosecond)
    ->ComputeStatistics("min", [](const auto &results) {
      return std::ranges::min(results);
    }) -> ComputeStatistics("max", [](const auto &results) {
  return std::ranges::max(results);
}) -> UseManualTime() -> DisplayAggregatesOnly(true) -> Repetitions(3);
} // namespace
} // namespace fcarouge::benchmark

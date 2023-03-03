/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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
#include "fcarouge/internal/utility.hpp"
#include "fcarouge/kalman.hpp"

#include <Eigen/Eigen>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>

namespace fcarouge::benchmark {
namespace {

template <auto Size> using vector = Eigen::Vector<float, Size>;

//! @benchmark Measure the prediction of the filter for different dimensions of
//! states and inputs with the Eigen linear algebra backend.
template <std::size_t StateSize, std::size_t InputSize>
void eigen_predict(::benchmark::State &state) {

  using kalman = kalman<vector<StateSize>, float, vector<InputSize>>;

  kalman filter;
  std::random_device random_device;
  std::mt19937 random_generator{random_device()};
  std::uniform_real_distribution<float> uniformly_distributed;

  for (auto _ : state) {

    typename kalman::input u;

    internal::for_constexpr<std::size_t{0}, InputSize, 1>(
        [&u, &uniformly_distributed, &random_generator](auto position) {
          u[position] = uniformly_distributed(random_generator);
        });

    ::benchmark::ClobberMemory();
    const auto start{clock::now()};

    filter.predict(u);

    ::benchmark::ClobberMemory();
    const auto end{clock::now()};

    state.SetIterationTime(std::chrono::duration<double>{end - start}.count());
  }
}

//! @todo Find a way to remove macros or find a different benchmark library that
//! doesn't use macros.
BENCHMARK(eigen_predict<${STATE_SIZE}, ${INPUT_SIZE}>)
    ->Name("eigen_predict_${STATE_SIZE}x1x${INPUT_SIZE}")
    ->Unit(::benchmark::kNanosecond)
    ->ComputeStatistics("min",
                        [](const auto &results) {
                          return std::ranges::min(results);
                        })
        -> ComputeStatistics("max",
                             [](const auto &results) {
                               return std::ranges::max(results);
                             }) -> UseManualTime()
            -> Complexity(::benchmark::oAuto) -> DisplayAggregatesOnly(true)
                -> Repetitions(100);

} // namespace
} // namespace fcarouge::benchmark

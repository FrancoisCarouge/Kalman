/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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

//! @file
//! @brief Benchmark of a sample's filter.
//!
//! @details Included ahead of the unchanged sample source, it declares the
//! `kalman` class template of the samples' namespace, which the sample's filter
//! declaration then names: the library filter, measuring its first prediction
//! and its first update. A measurement repeats the call, with the sample's
//! arguments, on a copy of the filter, and prints a line of the method name,
//! the median time per call in nanoseconds, its median absolute percentage
//! error, and the median instructions per call, or `-` without hardware
//! performance counters.

#ifndef FCAROUGE_BENCHMARK_HPP
#define FCAROUGE_BENCHMARK_HPP

#include "fcarouge/kalman.hpp"
#include "fcarouge/realtime.hpp"

#include <nanobench.h>

#include <format>
#include <print>
#include <string>
#include <utility>

namespace fcarouge::sample {
//! @brief The benchmark measures and prints in the sample's static
//! initialization: it opts out of the real-time verification for its lifetime.
inline const not_realtime opt_out;

//! @brief The library filter, measuring its first prediction and update.
template <typename Filter> class kalman : public fcarouge::kalman<Filter> {
public:
  using base = fcarouge::kalman<Filter>;
  using base::base;

  void predict(const auto &...arguments) {
    if (!std::exchange(predicted, true)) {
      bench("predict", static_cast<const base &>(*this),
            [&arguments...](base &filter) -> void {
              filter.predict(arguments...);
            });
    }
    base::predict(arguments...);
  }

  void update(const auto &...arguments) {
    if (!std::exchange(updated, true)) {
      bench("update", static_cast<const base &>(*this),
            [&arguments...](base &filter) -> void {
              filter.update(arguments...);
            });
    }
    base::update(arguments...);
  }

private:
  //! @brief Measures and prints the call, repeated on the filter.
  static void bench(const char *method, base filter, const auto &call) {
    using measure = ankerl::nanobench::Result::Measure;

    //! @brief Nanoseconds in a second, nanobench measuring elapsed time in
    //! seconds.
    constexpr double nanoseconds_per_second{1e9};

    //! @brief Percent in a ratio, nanobench reporting its median absolute
    //! percentage error as a ratio.
    constexpr double percent_per_ratio{100.};

    ankerl::nanobench::Bench runner;
    runner.output(nullptr).run(method, [&filter, &call] -> void {
      call(filter);
      ankerl::nanobench::doNotOptimizeAway(filter);
    });

    const auto &result{runner.results().front()};
    std::println(
        "{} {:.2f} {:.2f} {}", method,
        result.median(measure::elapsed) * nanoseconds_per_second,
        result.medianAbsolutePercentError(measure::elapsed) * percent_per_ratio,
        result.has(measure::instructions)
            ? std::format("{:.0f}", result.median(measure::instructions))
            : std::string{"-"});
  }

  //! @brief Whether the first prediction, update, of the filter type is
  //! measured.
  //! @{
  inline static bool predicted{false};
  inline static bool updated{false};
  //! @}
};

template <typename... Arguments>
kalman(Arguments... arguments) -> kalman<kf::deduce_filter<Arguments...>>;
} // namespace fcarouge::sample

#endif // FCAROUGE_BENCHMARK_HPP

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

#ifndef FCAROUGE_BENCHMARK_TPP
#define FCAROUGE_BENCHMARK_TPP

#include <nanobench.h>

#include <format>
#include <print>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge::sample {
namespace internal {
template <typename Filter>
constexpr benchmarker<Filter>::benchmarker(Filter &&filter)
    : Filter{std::move(filter)} {}

template <typename Filter>
constexpr void benchmarker<Filter>::predict(const auto &...arguments)
  requires requires(Filter &decorated) { decorated.predict(arguments...); }
{
  if (!std::exchange(predicted, true)) {
    bench("predict", static_cast<const Filter &>(*this),
          [values{std::tie(arguments...)}](Filter &filter) -> void {
            std::apply(
                [&filter](const auto &...value) -> void {
                  filter.predict(value...);
                },
                values);
          });
  }

  Filter::predict(arguments...);
}

template <typename Filter>
constexpr void benchmarker<Filter>::update(const auto &...arguments)
  requires requires(Filter &decorated) { decorated.update(arguments...); }
{
  if (!std::exchange(updated, true)) {
    bench("update", static_cast<const Filter &>(*this),
          [values{std::tie(arguments...)}](Filter &filter) -> void {
            std::apply(
                [&filter](const auto &...value) -> void {
                  filter.update(value...);
                },
                values);
          });
  }

  Filter::update(arguments...);
}

template <typename Filter>
void benchmarker<Filter>::bench(std::string_view method, Filter filter,
                                const auto &call) {
  using measure = ankerl::nanobench::Result::Measure;

  //! @brief Nanoseconds in a second, nanobench measuring elapsed time in
  //! seconds.
  constexpr double nanoseconds_per_second{1e9};

  //! @brief Percent in a ratio, nanobench reporting its median absolute
  //! percentage error as a ratio.
  constexpr double percent_per_ratio{100.};

  ankerl::nanobench::Bench runner;
  runner.output(nullptr).run(std::string{method}, [&filter, &call] -> void {
    call(filter);
    ankerl::nanobench::doNotOptimizeAway(filter);
  });

  const auto &result{runner.results().front()};
  std::println("{} {:.2f} {:.2f} {}", method,
               result.median(measure::elapsed) * nanoseconds_per_second,
               result.medianAbsolutePercentError(measure::elapsed) *
                   percent_per_ratio,
               result.has(measure::instructions)
                   ? std::format("{:.0f}", result.median(measure::instructions))
                   : std::string{"-"});
}
} // namespace internal

template <typename Filter>
  requires(std::is_class_v<Filter>)
[[nodiscard]] constexpr auto
operator|(Filter &&filter, [[maybe_unused]] const benchmarker &decorator) {
  return internal::benchmarker<Filter>{std::forward<Filter>(filter)};
}

template <typename Filter>
template <typename... Arguments>
kalman<Filter>::kalman(Arguments... arguments)
    : base{fcarouge::kalman<Filter>{arguments...} | benchmark} {}
} // namespace fcarouge::sample

#endif // FCAROUGE_BENCHMARK_TPP

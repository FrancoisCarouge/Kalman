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
//! @details Included ahead of the unchanged sample, its `kalman` shadows the
//! library's to decorate the sample's filter with `benchmark`.

#ifndef FCAROUGE_BENCHMARK_HPP
#define FCAROUGE_BENCHMARK_HPP

#include "fcarouge/kalman.hpp"
#include "fcarouge/realtime.hpp"

#include <string_view>
#include <type_traits>

namespace fcarouge::sample {
//! @brief Benchmark measuring and printing is not real-time safe by design.
const not_realtime opt_out;

namespace internal {
//! @brief Filter decorator benchmarking the first prediction and update.
//!
//! @details Repeats the call on a filter copy, then prints the method, median
//! nanoseconds, error percentage, and instructions, or `-` without counters.
template <typename Filter> class benchmarker : public Filter {
public:
  constexpr explicit benchmarker(Filter &&filter);

  constexpr void predict(const auto &...arguments)
    requires requires(Filter &decorated) { decorated.predict(arguments...); };

  constexpr void update(const auto &...arguments)
    requires requires(Filter &decorated) { decorated.update(arguments...); };

private:
  //! @brief Measures and prints the call, repeated on the filter.
  static void bench(std::string_view method, Filter filter, const auto &call);

  //! @brief Whether the first prediction, update, of the filter type is
  //! measured.
  //! @{
  inline static bool predicted{false};
  inline static bool updated{false};
  //! @}
};

} // namespace internal

struct benchmarker {};

//! @brief Filter decorator benchmarking the first prediction and update.
inline constexpr benchmarker benchmark;

//! @note The decorated filter is a prvalue or an xvalue, as for the library's
//! decorators.
template <typename Filter>
  requires(std::is_class_v<Filter>)
[[nodiscard]] constexpr auto operator|(Filter &&filter,
                                       const benchmarker &decorator);

//! @brief The library filter, decorated with `benchmark`.
//!
//! @details The sample's filter declaration names this class template rather
//! than the library's, and so is decorated without change.
template <typename Filter>
class kalman : public internal::benchmarker<fcarouge::kalman<Filter>> {
public:
  using base = internal::benchmarker<fcarouge::kalman<Filter>>;

  //! @note Forwarding rather than inheriting the constructor avoids an MSVC
  //! internal compiler error.
  template <typename... Arguments>
  explicit(false) kalman(Arguments... arguments);
};

template <typename... Arguments>
kalman(Arguments... arguments) -> kalman<kf::deduce_filter<Arguments...>>;
} // namespace fcarouge::sample

#include "benchmark.tpp"

#endif // FCAROUGE_BENCHMARK_HPP

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

#include "fcarouge/kalman.hpp"

#include <cstddef>

namespace fcarouge::test {
namespace {
template <typename Filter, typename... Arguments>
concept has_update = requires(Filter filter, Arguments... arguments) {
  filter.update(arguments...);
};

template <typename Filter, typename... Arguments>
concept has_predict = requires(Filter filter, Arguments... arguments) {
  filter.predict(arguments...);
};

template <typename Filter, std::size_t Position>
concept has_prediction_argument =
    requires(Filter filter) { filter.template predict<Position>(); };

template <typename Filter, typename... Values>
concept has_x =
    requires(Filter filter, Values... values) { filter.x(values...); };

struct nothing {};

//! @test Verifies the decorated filter accepts the calls the filter accepts,
//! and only those.
using filter = decltype(kalman{} | print);
using arguments = decltype(kalman{state{0.}, output<double>, input<double>,
                                  prediction_types<double>} |
                           print);

static_assert(not has_update<filter>);
static_assert(has_update<filter, double>);
static_assert(not has_update<filter, const char *>);
static_assert(not has_update<filter, double, double>);
static_assert(not has_predict<filter>);
static_assert(not has_prediction_argument<filter, 0>);
static_assert(has_x<filter>);
static_assert(has_x<const filter &>);
static_assert(has_x<filter, double>);
static_assert(not has_x<const filter &, double>);
static_assert(not has_x<filter, nothing>);
static_assert(has_predict<arguments, double, double>);
static_assert(not has_predict<arguments, nothing, double>);
static_assert(has_prediction_argument<arguments, 0>);
static_assert(not has_prediction_argument<arguments, 1>);
} // namespace
} // namespace fcarouge::test

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

namespace fcarouge::test {
namespace {
template <typename Filter, typename... Arguments>
concept has_predict = requires(Filter filter, Arguments... arguments) {
  filter.predict(arguments...);
};

struct nothing {};

//! @test Verifies the prediction member function is absent from the minimal
//! filter, there is no process to predict, from a filter called with an
//! unsupported number of arguments, or with values that do not initialize the
//! input without narrowing, present otherwise.
static_assert(not has_predict<decltype(kalman{})>);
static_assert(not has_predict<decltype(kalman{}), int, int, int>);
static_assert(has_predict<decltype(kalman{
                  state{0.}, output<double>, estimate_uncertainty{1.},
                  process_uncertainty{1.}, output_uncertainty{1.}})>);
static_assert(
    not has_predict<decltype(kalman{
                        state{0.}, output<double>, estimate_uncertainty{1.},
                        process_uncertainty{1.}, output_uncertainty{1.}}),
                    double>);
static_assert(
    has_predict<decltype(kalman{state{0.}, output<double>, input<double>}),
                double>);
static_assert(
    not has_predict<decltype(kalman{state{0.}, output<double>, input<double>}),
                    const char *>);
static_assert(
    not has_predict<decltype(kalman{state{0.}, output<double>, input<double>}),
                    nothing>);
static_assert(not has_predict<
              decltype(kalman{state{0.}, output<double>, input<double>}), int>);
static_assert(
    not has_predict<decltype(kalman{state{0.}, output<double>, input<double>}),
                    double, double>);
static_assert(
    not has_predict<decltype(kalman{state{0.}, output<double>, input<double>,
                                    prediction_types<double>}),
                    double>);
static_assert(
    has_predict<decltype(kalman{state{0.}, output<double>, input<double>,
                                prediction_types<double>}),
                double, double>);
static_assert(
    not has_predict<decltype(kalman{state{0.}, output<double>, input<double>,
                                    prediction_types<double>}),
                    nothing, double>);
} // namespace
} // namespace fcarouge::test

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

#include <concepts>
#include <tuple>

namespace fcarouge::test {
namespace {
//! @test Verifies the member types of the extended filter with callable process
//! uncertainty and output model functions of the state and the update and
//! prediction arguments, and a transition function.
using filter = decltype(kalman{
    state{0.}, output<double>, estimate_uncertainty{1.},
    process_uncertainty{[](const double &x, const float &step) -> double {
      return x * static_cast<double>(step);
    }},
    output_uncertainty{1.},
    output_model{
        [](const double &x, const int &scale) -> double { return x * scale; }},
    transition{[](const double &x,
                  [[maybe_unused]] const float &step) -> double { return x; }},
    observation{[](const double &x,
                   [[maybe_unused]] const int &scale) -> double { return x; }},
    update_types<int>, prediction_types<float>});

static_assert(std::same_as<filter::state, double>);
static_assert(std::same_as<filter::output, double>);
static_assert(std::same_as<filter::estimate_uncertainty, double>);
static_assert(std::same_as<filter::process_uncertainty, double>);
static_assert(std::same_as<filter::output_uncertainty, double>);
static_assert(std::same_as<filter::output_model, double>);
static_assert(std::same_as<filter::state_transition, double>);
static_assert(std::same_as<filter::gain, double>);
static_assert(std::same_as<filter::innovation, double>);
static_assert(std::same_as<filter::innovation_uncertainty, double>);
static_assert(std::same_as<filter::update_types, std::tuple<int>>);
static_assert(std::same_as<filter::prediction_types, std::tuple<float>>);
} // namespace
} // namespace fcarouge::test

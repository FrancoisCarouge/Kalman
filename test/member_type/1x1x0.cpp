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

namespace fcarouge::test {
namespace {
template <typename Filter>
concept has_input = requires { typename Filter::input; };

template <typename Filter>
concept has_input_control = requires { typename Filter::input_control; };

template <typename Filter>
concept has_output_model = requires { typename Filter::output_model; };

template <typename Filter>
concept has_prediction_types = requires { typename Filter::prediction_types; };

template <typename Filter>
concept has_process_uncertainty =
    requires { typename Filter::process_uncertainty; };

template <typename Filter>
concept has_state_transition = requires { typename Filter::state_transition; };

template <typename Filter>
concept has_update_types = requires { typename Filter::update_types; };

//! @test Verifies the member types of the minimal filter: present for its
//! characteristics, absent otherwise.
using filter = decltype(kalman{});

static_assert(std::same_as<filter::estimate_uncertainty, double>);
static_assert(std::same_as<filter::gain, double>);
static_assert(std::same_as<filter::innovation, double>);
static_assert(std::same_as<filter::innovation_uncertainty, double>);
static_assert(std::same_as<filter::output, double>);
static_assert(std::same_as<filter::output_uncertainty, double>);
static_assert(std::same_as<filter::state, double>);

static_assert(not has_input<filter>);
static_assert(not has_input_control<filter>);
static_assert(not has_output_model<filter>);
static_assert(not has_prediction_types<filter>);
static_assert(not has_process_uncertainty<filter>);
static_assert(not has_state_transition<filter>);
static_assert(not has_update_types<filter>);
} // namespace
} // namespace fcarouge::test

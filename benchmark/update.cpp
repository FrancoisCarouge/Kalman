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

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <nanobench.h>

#include <iostream>

namespace fcarouge::benchmark {
namespace {

using representation = float;

template <auto Size> using vector = column_vector<representation, Size>;

template <auto Size> auto state{fcarouge::state{vector<Size>{}}};

template <auto Size>
auto estimate_uncertainty{fcarouge::estimate_uncertainty{
    kalman_internal::ᴀʙᵀ<vector<Size>, vector<Size>>{}}};

template <auto Size>
auto process_uncertainty{fcarouge::process_uncertainty{
    kalman_internal::ᴀʙᵀ<vector<Size>, vector<Size>>{}}};

template <auto Size>
auto output_uncertainty{fcarouge::output_uncertainty{
    kalman_internal::ᴀʙᵀ<vector<Size>, vector<Size>>{}}};

template <auto Size>
auto state_transition{fcarouge::state_transition{
    kalman_internal::ᴀʙᵀ<vector<Size>, vector<Size>>{}}};

template <auto OutputSize, auto StateSize>
auto output_model{fcarouge::output_model{
    kalman_internal::ᴀʙᵀ<vector<OutputSize>, vector<StateSize>>{}}};

template <auto StateSize, auto InputSize>
auto input_control{fcarouge::input_control{
    kalman_internal::ᴀʙᵀ<vector<StateSize>, vector<InputSize>>{}}};

//! @test Verifies the copy constructor.
//!
//! Notes:
//! Make things work 0 states?
template <auto StateSize, auto OutputSize, auto InputSize>
[[maybe_unused]] auto benchmark{[] {
  kalman filter{state<StateSize>,                     // X
                output<vector<OutputSize>>,           // Z
                input<vector<InputSize>>,             // U
                estimate_uncertainty<StateSize>,      // P
                process_uncertainty<StateSize>,       // Q
                output_uncertainty<OutputSize>,       // R
                output_model<OutputSize, StateSize>,  // H
                state_transition<StateSize>,          // F
                input_control<StateSize, InputSize>,  // G
                update_types<>,
                prediction_types<>};
  ankerl::nanobench::Bench bench;
  const auto update_name{std::format("update_{}x{}", std::size_t{StateSize}, std::size_t{OutputSize})};

  bench.run(update_name, [&]() { filter.update(vector<OutputSize>{}); });

  return 0;
}};

[[maybe_unused]] auto run{benchmark<${STATE_SIZE}, ${IO_SIZE}, 1>()};
} // namespace
} // namespace fcarouge::benchmark

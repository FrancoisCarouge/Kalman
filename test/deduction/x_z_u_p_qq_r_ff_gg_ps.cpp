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
//! @brief A process noise covariance function of the state and the
//! prediction arguments.
struct process_noise {
  [[nodiscard]] static auto operator()(const double &x, const float &step)
      -> double {
    return x * static_cast<double>(step);
  }
};

//! @brief A state transition function of the input and the prediction
//! arguments.
struct propagation {
  [[nodiscard]] static auto operator()(const double &u, const float &step)
      -> double {
    return u * static_cast<double>(step);
  }
};

//! @brief An input control function of the prediction arguments.
struct control {
  [[nodiscard]] static auto operator()(const float &step) -> double {
    return static_cast<double>(step);
  }
};

//! @test Verifies the declaration deducing the filter with an input and
//! callable process uncertainty, state transition, and input control.
static_assert(
    std::same_as<
        decltype(kalman{state{0.}, output<double>, input<double>,
                        estimate_uncertainty{1.},
                        process_uncertainty{process_noise{}},
                        output_uncertainty{1.}, state_transition{propagation{}},
                        input_control{control{}}, prediction_types<float>}),
        kalman<kf::x_z_u_p_qq_r_ff_gg_ps<double, double, double, std::tuple<>,
                                         std::tuple<float>, process_noise,
                                         propagation, control>>>);
} // namespace
} // namespace fcarouge::test

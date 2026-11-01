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
#include "fcarouge/linalg.hpp"

#include <cstddef>

namespace fcarouge::benchmark {
namespace {
template <auto Size> using vector = column_vector<float, Size>;
template <auto Row, auto Column> using matrix = matrix<float, Row, Column>;

//! @brief Steps a gimbal attitude error filter, observing a velocity and a
//! heading error.
template <auto Policy>
const auto step{[](float rate, float north, float east, float down,
                   float heading) -> void {
  static constexpr auto jacobian{[](float angular_rate) -> matrix<9, 9> {
    matrix<9, 9> value{matrix<9, 9>::identity()};
    for (std::size_t axis{0}; axis < 3; ++axis) {
      value[axis, axis + 6] = -0.01F;
      value[axis + 3, axis] = angular_rate;
    }
    return value;
  }};

  static kalman filter{
      state{vector<9>{}},
      output<vector<4>>,
      estimate_uncertainty{0.01F * matrix<9, 9>::identity()},
      process_uncertainty{0.0001F * matrix<9, 9>::identity()},
      output_uncertainty{0.01F * matrix<4, 4>::identity()},
      output_model{[](const vector<9> &) -> matrix<4, 9> {
        matrix<4, 9> value;
        value[0, 3] = 1.F;
        value[1, 4] = 1.F;
        value[2, 5] = 1.F;
        value[3, 2] = 1.F;
        return value;
      }},
      state_transition{[](const vector<9> &, const float &angular_rate)
                           -> matrix<9, 9> { return jacobian(angular_rate); }},
      transition{[](const vector<9> &errors, const float &angular_rate)
                     -> vector<9> { return jacobian(angular_rate) * errors; }},
      observation{[](const vector<9> &errors) -> vector<4> {
        return vector<4>{errors[3], errors[4], errors[5], errors[2]};
      }},
      prediction_types<float>,
      Policy};

  filter.predict(rate);
  filter.update(north, east, down, heading);
}};
} // namespace
} // namespace fcarouge::benchmark

extern "C" {
[[gnu::flatten]] void joseph_form_step(float rate, float north, float east,
                                       float down, float heading) {
  fcarouge::benchmark::step<fcarouge::joseph_form>(rate, north, east, down,
                                                   heading);
}

[[gnu::flatten]] void simple_form_step(float rate, float north, float east,
                                       float down, float heading) {
  fcarouge::benchmark::step<fcarouge::simple_form>(rate, north, east, down,
                                                   heading);
}

[[gnu::flatten]] void information_form_step(float rate, float north, float east,
                                            float down, float heading) {
  fcarouge::benchmark::step<fcarouge::information_form>(rate, north, east, down,
                                                        heading);
}
}

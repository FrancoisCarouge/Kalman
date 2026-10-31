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

namespace fcarouge::benchmark {
namespace {
template <auto Size> using vector = column_vector<float, Size>;
template <auto Row, auto Column> using matrix = matrix<float, Row, Column>;

//! @brief Steps a thermal soaring filter, observing a variometer lift.
template <auto Policy>
const auto step{[](float drift_x, float drift_y, float position_x,
                   float position_y, float lift) -> void {
  static kalman filter{
      state{vector<4>{2.F, 80.F, 5.F, 5.F}},
      output<vector<1>>,
      estimate_uncertainty{{0.0049F, 0.F, 0.F, 0.F},
                           {0.F, 400.F, 0.F, 0.F},
                           {0.F, 0.F, 400.F, 0.F},
                           {0.F, 0.F, 0.F, 400.F}},
      process_uncertainty{{0.0001F, 0.F, 0.F, 0.F},
                          {0.F, 0.0001F, 0.F, 0.F},
                          {0.F, 0.F, 0.0001F, 0.F},
                          {0.F, 0.F, 0.F, 0.0001F}},
      output_uncertainty{0.2025F},
      output_model{[](const vector<4> &x, const float &px,
                      const float &py) -> matrix<1, 4> {
        return matrix<1, 4>{x[1], x[0], -2.F * (x[2] - px), -2.F * (x[3] - py)};
      }},
      transition{[](const vector<4> &x, const float &dx, const float &dy)
                     -> vector<4> { return x + vector<4>{0.F, 0.F, dx, dy}; }},
      observation{[](const vector<4> &x, const float &px,
                     const float &py) -> vector<1> {
        return vector<1>{(x[0] * x[1]) - ((x[2] - px) * (x[2] - px)) -
                         ((x[3] - py) * (x[3] - py))};
      }},
      update_types<float, float>,
      prediction_types<float, float>,
      Policy};

  filter.predict(drift_x, drift_y);
  filter.update(position_x, position_y, lift);
}};
} // namespace
} // namespace fcarouge::benchmark

extern "C" {
[[gnu::flatten]] void joseph_form_step(float drift_x, float drift_y,
                                       float position_x, float position_y,
                                       float lift) {
  fcarouge::benchmark::step<fcarouge::joseph_form>(drift_x, drift_y, position_x,
                                                   position_y, lift);
}

[[gnu::flatten]] void simple_form_step(float drift_x, float drift_y,
                                       float position_x, float position_y,
                                       float lift) {
  fcarouge::benchmark::step<fcarouge::simple_form>(drift_x, drift_y, position_x,
                                                   position_y, lift);
}

[[gnu::flatten]] void information_form_step(float drift_x, float drift_y,
                                            float position_x, float position_y,
                                            float lift) {
  fcarouge::benchmark::step<fcarouge::information_form>(
      drift_x, drift_y, position_x, position_y, lift);
}
}

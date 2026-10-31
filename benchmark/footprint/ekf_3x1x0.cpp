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

//! @brief Steps a beacon position filter, observing a squared range to a
//! beacon.
template <auto Policy>
const auto step{[](float north, float east, float down,
                   float squared_range) -> void {
  static kalman filter{
      state{vector<3>{5.F, 5.F, -1.75F}},
      output<vector<1>>,
      estimate_uncertainty{25.F * matrix<3, 3>::identity()},
      process_uncertainty{0.0001F * matrix<3, 3>::identity()},
      output_uncertainty{0.01F},
      output_model{[](const vector<3> &point,
                      const vector<3> &beacon) -> matrix<1, 3> {
        const vector<3> offset{point - beacon};
        return matrix<1, 3>{2.F * offset[0], 2.F * offset[1], 2.F * offset[2]};
      }},
      observation{
          [](const vector<3> &point, const vector<3> &beacon) -> vector<1> {
            const vector<3> offset{point - beacon};
            return vector<1>{(offset[0] * offset[0]) + (offset[1] * offset[1]) +
                             (offset[2] * offset[2])};
          }},
      update_types<vector<3>>,
      Policy};

  filter.predict();
  filter.update(vector<3>{north, east, down}, squared_range);
}};
} // namespace
} // namespace fcarouge::benchmark

extern "C" {
[[gnu::flatten]] void joseph_form_step(float north, float east, float down,
                                       float squared_range) {
  fcarouge::benchmark::step<fcarouge::joseph_form>(north, east, down,
                                                   squared_range);
}

[[gnu::flatten]] void simple_form_step(float north, float east, float down,
                                       float squared_range) {
  fcarouge::benchmark::step<fcarouge::simple_form>(north, east, down,
                                                   squared_range);
}

[[gnu::flatten]] void information_form_step(float north, float east, float down,
                                            float squared_range) {
  fcarouge::benchmark::step<fcarouge::information_form>(north, east, down,
                                                        squared_range);
}
}

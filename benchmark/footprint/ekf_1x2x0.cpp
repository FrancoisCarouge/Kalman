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

//! @brief Steps a terrain altitude filter, observing a range and a flow.
template <auto Policy>
const auto step{[](float range, float flow) -> void {
  static kalman filter{
      state{vector<1>{0.F}},
      output<vector<2>>,
      estimate_uncertainty{100.F},
      process_uncertainty{0.01F},
      output_uncertainty{{0.04F, 0.F}, {0.F, 0.0025F}},
      output_model{[](const vector<1> &altitude) -> matrix<2, 1> {
        return matrix<2, 1>{-1.F, altitude[0] / 50.F};
      }},
      observation{[](const vector<1> &altitude) -> vector<2> {
        return vector<2>{100.F - altitude[0],
                         altitude[0] * altitude[0] / 100.F};
      }},
      Policy};

  filter.predict();
  filter.update(range, flow);
}};
} // namespace
} // namespace fcarouge::benchmark

extern "C" {
[[gnu::flatten]] void joseph_form_step(float range, float flow) {
  fcarouge::benchmark::step<fcarouge::joseph_form>(range, flow);
}

[[gnu::flatten]] void simple_form_step(float range, float flow) {
  fcarouge::benchmark::step<fcarouge::simple_form>(range, flow);
}

[[gnu::flatten]] void information_form_step(float range, float flow) {
  fcarouge::benchmark::step<fcarouge::information_form>(range, flow);
}
}

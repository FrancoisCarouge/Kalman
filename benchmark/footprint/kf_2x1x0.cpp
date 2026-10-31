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

//! @brief Steps a stock price filter.
template <auto Policy>
const auto step{[](float price) -> void {
  static kalman filter{state{vector<2>{0.F, 0.F}},
                       output<vector<1>>,
                       estimate_uncertainty{{1000.F, 0.F}, {0.F, 1000.F}},
                       process_uncertainty{{0.25F, 0.F}, {0.F, 0.01F}},
                       output_uncertainty{1.F},
                       output_model{{1.F, 0.F}},
                       state_transition{{1.F, 1.F}, {0.F, 1.F}},
                       Policy};

  filter.predict();
  filter.update(price);
}};
} // namespace
} // namespace fcarouge::benchmark

extern "C" {
[[gnu::flatten]] void joseph_form_step(float price) {
  fcarouge::benchmark::step<fcarouge::joseph_form>(price);
}

[[gnu::flatten]] void simple_form_step(float price) {
  fcarouge::benchmark::step<fcarouge::simple_form>(price);
}

[[gnu::flatten]] void information_form_step(float price) {
  fcarouge::benchmark::step<fcarouge::information_form>(price);
}
}

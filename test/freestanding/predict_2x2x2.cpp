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

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies the prediction of a multi-dimension general filter on a
//! freestanding implementation: X = F.X + G.U and P = F.P.Fᵀ + Q.
static_assert([] -> bool {
  kalman filter{state{vector<2>{1., 2.}},
                output<vector<2>>,
                input<vector<2>>,
                estimate_uncertainty{{2., 1.}, {1., 2.}},
                process_uncertainty{{0.5, 0.}, {0., 0.5}},
                output_uncertainty{{1., 0.}, {0., 1.}},
                output_model{{1., 0.}, {0., 1.}},
                state_transition{{1., 1.}, {0., 1.}},
                input_control{{0.5, 0.}, {0., 1.}}};

  filter.predict(2., 4.);

  return filter.u() == vector<2>{2., 4.} && filter.x() == vector<2>{4., 6.} &&
         filter.p() == matrix<2, 2>{{6.5, 3.}, {3., 2.5}};
}());
} // namespace
} // namespace fcarouge::test

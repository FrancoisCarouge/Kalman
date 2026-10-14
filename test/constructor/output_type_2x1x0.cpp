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

#include <cassert>

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies the declared uncertainties and state transition initialize
//! their own characteristics of the multi-dimension filters with an output type
//! different from their state type: the undeclared process uncertainty and
//! output model keep their default values.
[[maybe_unused]] const auto test{[] -> int {
  kalman filter{state{vector<2>{5., 6.}}, output<double>,
                estimate_uncertainty{{2., 0.}, {0., 2.}},
                output_uncertainty{4.}};
  kalman another_filter{state{vector<2>{6., 7.}}, output<double>,
                        estimate_uncertainty{{2., 0.}, {0., 2.}},
                        output_uncertainty{4.},
                        state_transition{{1., 3.}, {0., 1.}}};

  assert((filter.x() == vector<2>{5., 6.}));
  assert((filter.p() == matrix<2, 2>{{2., 0.}, {0., 2.}}));
  assert((filter.q() == matrix<2, 2>{{0., 0.}, {0., 0.}}));
  assert(filter.r() == 4.);
  assert((filter.h() == matrix<1, 2>{{1., 0.}}));
  assert((filter.f() == matrix<2, 2>{{1., 0.}, {0., 1.}}));
  assert((another_filter.x() == vector<2>{6., 7.}));
  assert((another_filter.p() == matrix<2, 2>{{2., 0.}, {0., 2.}}));
  assert((another_filter.q() == matrix<2, 2>{{0., 0.}, {0., 0.}}));
  assert(another_filter.r() == 4.);
  assert((another_filter.h() == matrix<1, 2>{{1., 0.}}));
  assert((another_filter.f() == matrix<2, 2>{{1., 3.}, {0., 1.}}));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test

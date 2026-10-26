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

//! @test Verifies the runtime, constant, and mixed floors of a multi-dimension
//! filter raise each diagonal variance below the square of its standard
//! deviation floor, in state order, and keep the other elements.
[[maybe_unused]] const auto test{[] -> int {
  const auto declare{[](const auto &policy) -> auto {
    return kalman{
        state{vector<3>{0., 0., 0.}},
        output<vector<2>>,
        estimate_uncertainty{{0.25, 0.125, 0.}, {0.125, 4., 0.}, {0., 0., 1.}},
        process_uncertainty{{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}},
        output_uncertainty{{1., 0.}, {0., 1.}},
        output_model{{1., 0., 0.}, {0., 1., 0.}},
        state_transition{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}},
        policy};
  }};

  auto runtime{declare(floors{1., 1., 2.})};
  auto constant{declare(floors{cw<1.>, cw<1.>, cw<2.>})};
  auto mixed{declare(floors{1., cw<1.>, 2.})};

  runtime.predict();
  constant.predict();
  mixed.predict();

  for (int row{0}; row < 3; ++row) {
    for (int column{0}; column < 3; ++column) {
      assert(runtime.p()(row, column) == constant.p()(row, column));
      assert(runtime.p()(row, column) == mixed.p()(row, column));
    }
  }

  assert(runtime.p()(0, 0) == 1.);
  assert(runtime.p()(1, 1) == 4.);
  assert(runtime.p()(2, 2) == 4.);
  assert(runtime.p()(0, 1) == 0.125 && runtime.p()(1, 0) == 0.125);
  assert(runtime.p()(0, 2) == 0. && runtime.p()(2, 1) == 0.);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
